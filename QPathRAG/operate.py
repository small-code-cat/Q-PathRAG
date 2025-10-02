import asyncio
import json
import re

import torch
from tqdm.asyncio import tqdm as tqdm_async
from typing import Union
from scipy.special import softmax
from collections import Counter, defaultdict
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import warnings
from .utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    process_combine_contexts,
    compute_args_hash,
    handle_cache,
    save_to_cache,
    CacheData,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS

bge_model = SentenceTransformer("/mnt/data/xkj/BAAI/bge-m3")

# sentence_model = SentenceTransformer('/mnt/data/xkj/sentence-transformers/all-MiniLM-L6-v2')  # 可换为 DeepSeek/Cohere 等
#
# # 加载 SapBERT 模型
# bio_tokenizer = AutoTokenizer.from_pretrained("/mnt/data/xkj/cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
# bio_model = AutoModel.from_pretrained("/mnt/data/xkj/cambridgeltl/SapBERT-from-PubMedBERT-fulltext").to(sentence_model.device)
#
# his_tokenizer = AutoTokenizer.from_pretrained("/mnt/data/xkj/dbmdz/bert-medium-historic-multilingual-cased")
# his_model = AutoModel.from_pretrained("/mnt/data/xkj/dbmdz/bert-medium-historic-multilingual-cased").to(sentence_model.device)
#
# agri_tokenizer = AutoTokenizer.from_pretrained("/mnt/data/xkj/recobo/agriculture-bert-uncased")
# agri_model = AutoModel.from_pretrained("/mnt/data/xkj/recobo/agriculture-bert-uncased").to(sentence_model.device)
#
# legal_tokenizer = AutoTokenizer.from_pretrained("/mnt/data/xkj/nlpaueb/legal-bert-base-uncased")
# legal_model = AutoModel.from_pretrained("/mnt/data/xkj/nlpaueb/legal-bert-base-uncased").to(sentence_model.device)

def chunking_by_token_size(
    content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
):
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens: 
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
        language=language,
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
   
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
   
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])

    edge_keywords = clean_str(record_attributes[4])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        keywords=edge_keywords,
        source_id=edge_source_id,
    )


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_entity_types = []
    already_source_ids = []
    already_description = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entity_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entity_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []
    already_description = []
    already_keywords = []

    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_keywords.extend(
            split_string_by_multi_markers(already_edge["keywords"], [GRAPH_FIELD_SEP])
        )

    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    keywords = GRAPH_FIELD_SEP.join(
        sorted(set([dp["keywords"] for dp in edges_data] + already_keywords))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    for need_insert_id in [src_id, tgt_id]:
        if not (await knowledge_graph_inst.has_node(need_insert_id)):
            await knowledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    description = await _handle_entity_relation_summary(
        f"({src_id}, {tgt_id})", description, global_config
    )
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight,
            description=description,
            keywords=keywords,
            source_id=source_id,
        ),
    )

    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        description=description,
        keywords=keywords,
    )

    return edge_data


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    # time.sleep(20)
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())
  
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )
    entity_types = global_config["addon_params"].get(
        "entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"]
    )
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["entity_extraction_examples"]):
        examples = "\n".join(
            PROMPTS["entity_extraction_examples"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["entity_extraction_examples"])

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        language=language,
    )
  
    examples = examples.format(**example_context_base)

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        examples=examples,
        language=language,
    )

    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        hint_prompt = entity_extract_prompt.format(
            **context_base, input_text="{input_text}"
        ).format(**context_base, input_text=content)

        final_result = await use_llm_func(hint_prompt)
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await use_llm_func(
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    results = []
    for result in tqdm_async(
        asyncio.as_completed([_process_single_content(c) for c in ordered_chunks]),
        total=len(ordered_chunks),
        desc="Extracting entities from chunks",
        unit="chunk",
    ):
        results.append(await result)

    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[k].extend(v)
    logger.info("Inserting entities into storage...")
    all_entities_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [
                _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
                for k, v in maybe_nodes.items()
            ]
        ),
        total=len(maybe_nodes),
        desc="Inserting entities",
        unit="entity",
    ):
        all_entities_data.append(await result)

    logger.info("Inserting relationships into storage...")
    all_relationships_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [
                _merge_edges_then_upsert(
                    k[0], k[1], v, knowledge_graph_inst, global_config
                )
                for k, v in maybe_edges.items()
            ]
        ),
        total=len(maybe_edges),
        desc="Inserting relationships",
        unit="relationship",
    ):
        all_relationships_data.append(await result)

    if not len(all_entities_data) and not len(all_relationships_data):
        logger.warning(
            "Didn't extract any entities and relationships, maybe your LLM is not working"
        )
        return None

    if not len(all_entities_data):
        logger.warning("Didn't extract any entities")
    if not len(all_relationships_data):
        logger.warning("Didn't extract any relationships")

    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)

    if relationships_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                "src_id": dp["src_id"],
                "tgt_id": dp["tgt_id"],
                "content": dp["keywords"]
                + dp["src_id"]
                + dp["tgt_id"]
                + dp["description"],
            }
            for dp in all_relationships_data
        }
        await relationships_vdb.upsert(data_for_vdb)

    return knowledge_graph_inst



async def kg_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
) -> str:

    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash(query_param.mode, query)
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode
    )
    if cached_response is not None:
        return cached_response

    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["keywords_extraction_examples"]):
        examples = "\n".join(
            PROMPTS["keywords_extraction_examples"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["keywords_extraction_examples"])
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    if query_param.mode not in ["hybrid"]:
        logger.error(f"Unknown mode {query_param.mode} in kg_query")
        return PROMPTS["fail_response"]


    kw_prompt_temp = PROMPTS["keywords_extraction"]
    kw_prompt = kw_prompt_temp.format(query=query, examples=examples, language=language)
    result = await use_model_func(kw_prompt, keyword_extraction=True)
    logger.info("kw_prompt result:")
    print(result)
    try:

        match = re.search(r"\{.*\}", result, re.DOTALL)
        if match:
            result = match.group(0)
            keywords_data = json.loads(result)

            hl_keywords = keywords_data.get("high_level_keywords", [])
            ll_keywords = keywords_data.get("low_level_keywords", [])
        else:
            logger.error("No JSON-like structure found in the result.")
            return PROMPTS["fail_response"]


    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e} {result}")
        return PROMPTS["fail_response"]


    if hl_keywords == [] and ll_keywords == []:
        logger.warning("low_level_keywords and high_level_keywords is empty")
        return PROMPTS["fail_response"]
    if ll_keywords == [] and query_param.mode in ["hybrid"]:
        logger.warning("low_level_keywords is empty")
        return PROMPTS["fail_response"]
    else:
        ll_keywords = ", ".join(ll_keywords)
    if hl_keywords == [] and query_param.mode in ["hybrid"]:
        logger.warning("high_level_keywords is empty")
        return PROMPTS["fail_response"]
    else:
        hl_keywords = ", ".join(hl_keywords)


    keywords = [ll_keywords, hl_keywords]
    context= await _build_query_context(
        keywords,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
        global_config,
        query
    )

    

    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]
    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    if query_param.only_need_prompt:
        return sys_prompt
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )


    await save_to_cache(
        hashing_kv,
        CacheData(
            args_hash=args_hash,
            content=response,
            prompt=query,
            quantized=quantized,
            min_val=min_val,
            max_val=max_val,
            mode=query_param.mode,
        ),
    )
    return response


async def _build_query_context(
    query: list,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config,
    question
):
    ll_entities_context, ll_relations_context, ll_text_units_context = "", "", ""
    hl_entities_context, hl_relations_context, hl_text_units_context = "", "", ""

    ll_kewwords, hl_keywrds = query[0], query[1]
    if query_param.mode in ["local", "hybrid"]:
        if ll_kewwords == "":
            ll_entities_context, ll_relations_context, ll_text_units_context = (
                "",
                "",
                "",
            )
            warnings.warn(
                "Low Level context is None. Return empty Low entity/relationship/source"
            )
            query_param.mode = "global"
        else:
            (
                ll_entities_context,
                ll_relations_context,
                ll_text_units_context,
            ) = await _get_node_data(
                ll_kewwords,
                knowledge_graph_inst,
                entities_vdb,
                text_chunks_db,
                query_param,
                global_config,
                question
            )
    if query_param.mode in ["hybrid"]:
        if hl_keywrds == "":
            hl_entities_context, hl_relations_context, hl_text_units_context = (
                "",
                "",
                "",
            )
            warnings.warn(
                "High Level context is None. Return empty High entity/relationship/source"
            )
            query_param.mode = "local"
        else:
            (
                hl_entities_context,
                hl_relations_context,
                hl_text_units_context,
            ) = await _get_edge_data(
                hl_keywrds,
                knowledge_graph_inst,
                relationships_vdb,
                text_chunks_db,
                query_param,
            )
            if (
                hl_entities_context == ""
                and hl_relations_context == ""
                and hl_text_units_context == ""
            ):
                logger.warn("No high level context found. Switching to local mode.")
                query_param.mode = "local"
    if query_param.mode == "hybrid":
        entities_context, relations_context, text_units_context = combine_contexts(
            [hl_entities_context, hl_relations_context],
            [ll_entities_context, ll_relations_context],
            [hl_text_units_context, ll_text_units_context],
        )


    return f"""
-----global-information-----
-----high-level entity information-----
```csv
{hl_entities_context}
```
-----high-level relationship information-----
```csv
{hl_relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
-----local-information-----
-----low-level entity information-----
```csv
{ll_entities_context}
```
-----low-level relationship information-----
```csv
{ll_relations_context}
```
"""

async def _get_node_data(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config,
    question
):
    use_model_func = global_config["llm_model_func"]
    embedding_func = global_config['embedding_func']
    results = await entities_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return "", "", ""

    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")


    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]  
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )

    # 使用大模型对问题进行打分
    # response = await use_model_func(PROMPTS['question_classify'].format(question=question))
    # parsed_response = {k.strip().lower(): float(v.strip()) if k.strip().lower() == "confidence" else v.strip()
    #           for k, v in (line.split(":", 1) for line in response.strip().split("\n"))}


    use_relations= await _find_most_related_edges_from_entities3(
        node_datas, query_param, knowledge_graph_inst, question, use_model_func, embedding_func
    )

    logger.info(
        f"Local query uses {len(node_datas)} entites, {len(use_relations)} relations, {len(use_text_units)} text units"
    )


    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    relations_section_list=[["id","context"]]
    for i,e in enumerate(use_relations):
        relations_section_list.append([i,e])
    relations_context=list_of_list_to_csv(relations_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    
    return entities_context,relations_context,text_units_context


async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])

    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )


    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None and "source_id" in v  
    }

    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id not in all_text_units_lookup:
                all_text_units_lookup[c_id] = {
                    "data": await text_chunks_db.get_by_id(c_id),
                    "order": index,
                    "relation_counts": 0,
                }

            if this_edges:
                for e in this_edges:
                    if (
                        e[1] in all_one_hop_text_units_lookup
                        and c_id in all_one_hop_text_units_lookup[e[1]]
                    ):
                        all_text_units_lookup[c_id]["relation_counts"] += 1


    all_text_units = [
        {"id": k, **v}
        for k, v in all_text_units_lookup.items()
        if v is not None and v.get("data") is not None and "content" in v["data"]
    ]

    if not all_text_units:
        logger.warning("No valid text units found")
        return []

    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )

    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    all_text_units = [t["data"] for t in all_text_units]
    return all_text_units

async def _get_edge_data(
    keywords,
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    results = await relationships_vdb.query(keywords, top_k=query_param.top_k)

    if not len(results):
        return "", "", ""

    edge_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(r["src_id"], r["tgt_id"]) for r in results]
    )

    if not all([n is not None for n in edge_datas]):
        logger.warning("Some edges are missing, maybe the storage is damaged")
    edge_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(r["src_id"], r["tgt_id"]) for r in results]
    )
    edge_datas = [
        {"src_id": k["src_id"], "tgt_id": k["tgt_id"], "rank": d, **v}
        for k, v, d in zip(results, edge_datas, edge_degree)
        if v is not None
    ]
    edge_datas = sorted(
        edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    edge_datas = truncate_list_by_token_size(
        edge_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )

    use_entities = await _find_most_related_entities_from_relationships(
        edge_datas, query_param, knowledge_graph_inst
    )
    use_text_units = await _find_related_text_unit_from_relationships(
        edge_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    logger.info(
        f"Global query uses {len(use_entities)} entites, {len(edge_datas)} relations, {len(use_text_units)} text units"
    )

    relations_section_list = [
        ["id", "source", "target", "description", "keywords", "weight", "rank"]
    ]
    for i, e in enumerate(edge_datas):
        relations_section_list.append(
            [
                i,
                e["src_id"],
                e["tgt_id"],
                e["description"],
                e["keywords"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(use_entities):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return entities_context, relations_context, text_units_context


async def _find_most_related_entities_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    entity_names = []
    seen = set()

    for e in edge_datas:
        if e["src_id"] not in seen:
            entity_names.append(e["src_id"])
            seen.add(e["src_id"])
        if e["tgt_id"] not in seen:
            entity_names.append(e["tgt_id"])
            seen.add(e["tgt_id"])

    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(entity_name) for entity_name in entity_names]
    )

    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(entity_name) for entity_name in entity_names]
    )
    node_datas = [
        {**n, "entity_name": k, "rank": d}
        for k, n, d in zip(entity_names, node_datas, node_degrees)
    ]

    node_datas = truncate_list_by_token_size(
        node_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_local_context,
    )

    return node_datas


async def _find_related_text_unit_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in edge_datas
    ]
    all_text_units_lookup = {}

    for index, unit_list in enumerate(text_units):
        for c_id in unit_list:
            if c_id not in all_text_units_lookup:
                chunk_data = await text_chunks_db.get_by_id(c_id)

                if chunk_data is not None and "content" in chunk_data:
                    all_text_units_lookup[c_id] = {
                        "data": chunk_data,
                        "order": index,
                    }

    if not all_text_units_lookup:
        logger.warning("No valid text chunks found")
        return []

    all_text_units = [{"id": k, **v} for k, v in all_text_units_lookup.items()]
    all_text_units = sorted(all_text_units, key=lambda x: x["order"])


    valid_text_units = [
        t for t in all_text_units if t["data"] is not None and "content" in t["data"]
    ]

    if not valid_text_units:
        logger.warning("No valid text chunks after filtering")
        return []

    truncated_text_units = truncate_list_by_token_size(
        valid_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    all_text_units: list[TextChunkSchema] = [t["data"] for t in truncated_text_units]

    return all_text_units


def combine_contexts(entities, relationships, sources):

    hl_entities, ll_entities = entities[0], entities[1]
    hl_relationships, ll_relationships = relationships[0], relationships[1]
    hl_sources, ll_sources = sources[0], sources[1]

    combined_entities = process_combine_contexts(hl_entities, ll_entities)

    combined_relationships = process_combine_contexts(
        hl_relationships, ll_relationships
    )

    combined_sources = process_combine_contexts(hl_sources, ll_sources)

    return combined_entities, combined_relationships, combined_sources


import networkx as nx
from collections import defaultdict
async def find_paths_and_edges_with_stats(graph, source_nodes, target_nodes):

    result = defaultdict(lambda: {"paths": [], "edges": set()})
    path_stats = {"1-hop": 0, "2-hop": 0, "3-hop": 0}   
    one_hop_paths = []
    two_hop_paths = []
    three_hop_paths = []

    async def dfs(current, target, path, depth):

        if depth > 3: 
            return
        if current == target: 
            result[(path[0], target)]["paths"].append(list(path))
            for u, v in zip(path[:-1], path[1:]):
                result[(path[0], target)]["edges"].add(tuple(sorted((u, v))))
            if depth == 1:
                path_stats["1-hop"] += 1
                one_hop_paths.append(list(path))
            elif depth == 2:
                path_stats["2-hop"] += 1
                two_hop_paths.append(list(path))
            elif depth == 3:
                path_stats["3-hop"] += 1
                three_hop_paths.append(list(path))
            return
        neighbors = graph.neighbors(current) 
        for neighbor in neighbors:
            if neighbor not in path:  
                await dfs(neighbor, target, path + [neighbor], depth + 1)

    for node1 in source_nodes:
        for node2 in target_nodes:
            if node1 != node2:
                await dfs(node1, node2, [node1], 0)

    for key in result:
        result[key]["edges"] = list(result[key]["edges"])

    return dict(result), path_stats , one_hop_paths, two_hop_paths, three_hop_paths
def bfs_weighted_paths(G, path, source, target, threshold, alpha):
    results = [] 
    edge_weights = defaultdict(float)  
    node = source
    follow_dict = {}

    for p in path:
        for i in range(len(p) - 1):  
            current = p[i]
            next_num = p[i + 1]

            if current in follow_dict:
                follow_dict[current].add(next_num)
            else:
                follow_dict[current] = {next_num}

    for neighbor in follow_dict[node]:
        edge_weights[(node, neighbor)] += 1/len(follow_dict[node])

        if neighbor == target:
            results.append(([node, neighbor]))
            continue
        
        if edge_weights[(node, neighbor)] > threshold:

            for second_neighbor in follow_dict[neighbor]:
                weight = edge_weights[(node, neighbor)] * alpha / len(follow_dict[neighbor])
                edge_weights[(neighbor, second_neighbor)] += weight

                if second_neighbor == target:
                    results.append(([node, neighbor, second_neighbor]))
                    continue

                if edge_weights[(neighbor, second_neighbor)] > threshold:    

                    for third_neighbor in follow_dict[second_neighbor]:
                        weight = edge_weights[(neighbor, second_neighbor)] * alpha / len(follow_dict[second_neighbor]) 
                        edge_weights[(second_neighbor, third_neighbor)] += weight

                        if third_neighbor == target :
                            results.append(([node, neighbor, second_neighbor, third_neighbor]))
                            continue
    path_weights = []
    for p in path:
        path_weight = 0
        for i in range(len(p) - 1):
            edge = (p[i], p[i + 1])
            path_weight += edge_weights.get(edge, 0)  
        path_weights.append(path_weight/(len(p)-1))

    combined = [(p, w) for p, w in zip(path, path_weights)]

    return combined

# ========== Step 1: 初始化关系向量 ==========
def compute_relation_embeddings(G):
    relation_set = set()
    for _, _, data in G.edges(data=True):
        if "relation" in data:
            relation_set.add(data["relation"])
    relation_list = list(relation_set)
    relation_embedding = sentence_model.encode(relation_list, show_progress_bar=False)
    rel_embeddings = {
        rel:rel_embedding for rel, rel_embedding in zip(relation_list, relation_embedding)
    }
    return rel_embeddings

# ========== Step 2: 语义相似度引导路径扩展 ==========
def semantic_expand_nx(
    G,
    start_entity,
    question_text,
    rel_embeddings,
    max_hops=2,
    sim_threshold=0.3,
    top_k_per_hop=5
):
    question_vec = sentence_model.encode(question_text, show_progress_bar=False)
    paths = [[start_entity]]  # initial path

    for hop in range(max_hops):
        new_paths = []
        for path in paths:
            current = path[-1]
            if current not in G:
                continue
            neighbors = []
            for neighbor in G.neighbors(current):
                if neighbor in path:
                    continue  # 避免环
                edge_data = G.get_edge_data(current, neighbor)
                if not edge_data:
                    continue
                rel = edge_data.get("relation")
                if rel not in rel_embeddings:
                    continue
                rel_vec = rel_embeddings[rel]
                sim = cosine_similarity([question_vec], [rel_vec])[0][0]
                if sim >= sim_threshold:
                    neighbors.append((sim, rel, neighbor))
            # 选择最相关的 Top-K 扩展
            neighbors = sorted(neighbors, reverse=True)[:top_k_per_hop]
            for sim, rel, neighbor in neighbors:
                new_paths.append(path + [rel, neighbor])
        paths = new_paths
        if not paths:
            break
    return paths

async def format_kg_path_description(path, knowledge_graph_inst):
    """
    输入路径: ["Einstein", "hasStudent", "Bohr", "hasStudent", "Heisenberg"]
    输出路径描述语句
    """
    parts = []  # 存放每段 entity + edge 的解释文本

    for i in range(0, len(path)-1, 1):
        e1, e2 = path[i], path[i + 1]
        edge = await knowledge_graph_inst.get_edge(e1, e2) or await knowledge_graph_inst.get_edge(e2, e1)
        rel = edge['keywords']

        # 构造边描述
        edge_text = f"through edge ({rel}) to connect {e1} and {e2}."

        # 获取实体信息
        node1 = await knowledge_graph_inst.get_node(e1)
        node2 = await knowledge_graph_inst.get_node(e2)

        node1_text = f"The entity {e1} is a {node1['entity_type']} with the description({node1['description']})"
        node2_text = f"The entity {e2} is a {node2['entity_type']} with the description({node2['description']})"

        parts.append(node1_text + edge_text + node2_text)

    # 将所有段用 and 连接
    full_description = " and ".join(parts)
    return full_description

async def path_to_natural_sentence2(path, knowledge_graph_inst):
    """
    输入: ["Einstein", "Bohr", "Heisenberg"]
    输出: The entity Einstein is a physicist with the description (...)
          had a relation 'hasStudent' with Bohr,
          who in turn had a 'hasStudent' relation with Heisenberg.
          Heisenberg is a physicist with the description (...).
    """
    if not path:
        return ""

    text_parts = []

    # 起始实体描述
    start = path[0]
    start_info = await knowledge_graph_inst.get_node(start)
    start_type = start_info.get("entity_type", "entity")
    start_desc = start_info.get("description", "")
    text_parts.append(f"{start}, a {start_type}, {start_desc}")

    # 中间路径段（边 + 下一个实体）
    for i in range(len(path) - 1):
        src = path[i]
        tgt = path[i + 1]

        edge = await knowledge_graph_inst.get_edge(src, tgt) or await knowledge_graph_inst.get_edge(tgt, src)
        rel = edge['keywords'] if edge else "unknown relation"

        if i == 0:
            text_parts.append(f" had a relation '{rel}' with {tgt}")
        else:
            text_parts.append(f", who in turn had a '{rel}' relation with {tgt}")

    # 终点实体附加描述
    last = path[-1]
    if last != start:  # 避免首尾相同重复描述
        last_info = await knowledge_graph_inst.get_node(last)
        entity_type = last_info.get("entity_type", "entity")
        description = last_info.get("description", "")
        text_parts.append(f". {last} is a {entity_type}, {description}.")

    return " ".join(text_parts)

def deduplicate_paths_directionless(paths):
    seen = set()
    unique = []
    for path in paths:
        fwd = tuple(path)
        rev = tuple(reversed(path))
        if fwd not in seen and rev not in seen:
            seen.add(fwd)
            unique.append(path)
    return unique

def extract_k_hop_subgraph(G, seeds, k=3):
    """
    G: NetworkX 图（无向或有向）
    seeds: list[str]，起始实体节点（可多个）
    k: int，最大跳数
    return: 子图（NetworkX 子图对象）
    """
    visited = set(seeds)
    frontier = set(seeds)

    for _ in range(k):
        next_frontier = set()
        for node in frontier:
            neighbors = set(G.neighbors(node))
            next_frontier |= neighbors
        next_frontier -= visited  # 避免回环
        visited |= next_frontier
        frontier = next_frontier

    return G.subgraph(visited).copy()

def filter_subgraph_by_semantics(G_sub, question_text, rel_embeddings, sim_threshold=0.4):
    """
    对子图进行语义边过滤，并移除孤立节点
    参数：
        G_sub: 输入子图（NetworkX 图）
        question_text: 问题句
        model: Sentence-BERT 模型
        rel_embeddings: dict[str, vector]，relation 的语义向量
        sim_threshold: 最小相似度阈值
    返回：
        G_filtered: 过滤后的子图（仅包含语义相关边，去除孤立节点）
    """
    question_vec = sentence_model.encode(question_text, show_progress_bar=False)
    G_filtered = nx.Graph()

    for u, v, data in G_sub.edges(data=True):
        rel = data.get("relation")
        if rel in rel_embeddings:
            rel_vec = rel_embeddings[rel]
            sim = cosine_similarity([question_vec], [rel_vec])[0][0]
            if sim >= sim_threshold:
                G_filtered.add_edge(u, v, **data)

    return G_filtered


def allocate_paths_from_softmax(weights, total_paths):
    ideal_counts = weights * total_paths

    # Step 3: 初始整数分配 = 向下取整
    int_counts = np.floor(ideal_counts).astype(int)
    remaining = total_paths - int_counts.sum()

    # Step 4: 按误差从大到小排序，分配剩余路径
    residuals = ideal_counts - int_counts
    for idx in np.argsort(-residuals):
        if remaining <= 0:
            break
        int_counts[idx] += 1
        remaining -= 1

    return int_counts

def select_top_paths_per_subq_no_repeat(similarity_matrix, path_list, allocation):
    """
    为每个子问题选择 top-k 路径，避免路径重复（每条路径只能属于一个子问题）

    参数：
    - similarity_matrix: ndarray (num_paths, num_subqs)，每个路径与每个子问题的相似度
    - path_list: List[str]，路径文本列表
    - allocation: List[int] or ndarray，表示每个子问题需要几条路径

    返回：
    - selected: Dict[subq_idx, List[Tuple[path_idx, score, path_text]]]
    """
    selected = {}
    used_indices = set()
    num_paths, num_subqs = similarity_matrix.shape

    for subq_idx, k in enumerate(allocation):
        sims = similarity_matrix[:, subq_idx]

        # 获取未被使用的路径索引和相应分数
        path_scores = [(i, sims[i]) for i in range(num_paths) if i not in used_indices]
        # 按分数排序，选出 top-k
        top_k = sorted(path_scores, key=lambda x: -x[1])[:k]

        selected[subq_idx] = [(i, score, path_list[i]) for i, score in top_k]
        used_indices.update(i for i, _ in top_k)

    return selected

def get_umls_embeddings(text, tokenizer, model):
    try:
        with torch.no_grad():
            inputs = tokenizer(text, truncation=True, max_length=512, padding=True, return_tensors="pt").to(bio_model.device)
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()  # [num_concepts, dim]
    except Exception as e:
        batch_size = 64
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(text), batch_size):
                batch = text[i:i + batch_size]
                inputs = tokenizer(batch, truncation=True, padding=True, max_length=512, return_tensors="pt").to(bio_model.device)
                outputs = model(**inputs)
                embeddings.append(outputs.last_hidden_state.mean(dim=1))
                del inputs
                del outputs
                torch.cuda.empty_cache()
        return torch.cat(embeddings, dim=0).detach().cpu().numpy()

def get_embedding(text):
    # return get_umls_embeddings(text, bio_tokenizer, bio_model)
    # return get_umls_embeddings(text, agri_tokenizer, agri_model)
    return get_umls_embeddings(text, legal_tokenizer, legal_model)
    # return get_umls_embeddings(text, his_tokenizer, his_model)
    # return sentence_model.encode(text, show_progress_bar=False)

def is_valid_embedding_string(text: str) -> bool:
    if not isinstance(text, str):
        return False
    text = text.strip()
    if text == "":
        return False
    return True

async def embed_all_texts(all_texts, embedding_func, batch_size=256):
    all_embeddings = []
    all_texts = [text for text in all_texts if is_valid_embedding_string(text)]
    for i in range(0, len(all_texts), batch_size):
        batch = all_texts[i:i+batch_size]
        batch_embeddings = await embedding_func(batch, dimensions=512)
        all_embeddings.append(batch_embeddings)
    return np.vstack(all_embeddings)

def get_target_nodes(G, question_embedding, similaity_number=25):
    all_nodes = list(G.nodes)
    all_node_embedding = bge_model.encode(all_nodes, show_progress_bar=False)
    all_node_subquestion_similarity = bge_model.similarity(all_node_embedding, question_embedding[1:]).cpu().numpy()
    selected_nodes = set()
    for subq_idx in range(len(question_embedding[1:])):
        sims = all_node_subquestion_similarity[:, subq_idx]
        topk_indices = sims.argsort()[::-1][:similaity_number]  # 降序取 topk
        for idx in topk_indices:
            selected_nodes.add(all_nodes[idx])  # 添加节点名称
    selected_nodes = list(selected_nodes)
    return selected_nodes

async def get_question_embedding(use_model_func, question_text):
    response = await use_model_func(PROMPTS['question_decomposition'].format(question=question_text))
    sub_questions = re.findall(r'\d+\.\s+(.*?)(?=\n\d+\.|\Z)', response.strip(), re.DOTALL)

    # question_embedding = await embed_all_texts([question_text]+sub_questions, embedding_func)
    # question_similarity = cosine_similarity(question_embedding[0][None], question_embedding[1:])[0]
    question_embedding = bge_model.encode([question_text] + sub_questions, show_progress_bar=False)
    return question_embedding

async def extract_and_rank_natural_paths(
    G,
    source_nodes,
    question_text,
    knowledge_graph_inst,
    use_model_func,
    embedding_func,
    max_hops=3,
    top_k_paths=10
):
    """
    从图中提取结构路径 → 转为自然语言句子 → 按与问题的语义相似度排序

    参数：
        G: NetworkX 图（无向图或 MultiDiGraph）
        question_text: 待匹配问题文本
        knowledge_graph_inst: 提供 get_node(entity) 异步接口
        max_hops: 最多路径跳数
        top_k_paths: 返回 top-k 最相关路径

    返回：
        List[(score, path: list, natural_sentence: str)]
    """
    question_embedding = await get_question_embedding(use_model_func, question_text)
    question_similarity = bge_model.similarity(question_embedding[0][None], question_embedding[1:])[0].cpu().numpy()
    question_importance = softmax(question_similarity)
    path_count = allocate_paths_from_softmax(question_importance, top_k_paths)
    all_results = []
    added_entities = {}

    def path_to_natural_sentence3(path, edge_list, entity_info_map):
        if not path or not edge_list:
            return ""

        parts = []  # 存放每一段 entity + relation + entity 的自然语言描述

        for i in range(len(edge_list)):
            src = path[i]
            tgt = path[i + 1]
            rel = edge_list[i]

            # 获取实体信息
            src_text = entity_info_map.get(src, src)
            tgt_text = entity_info_map.get(tgt, tgt)

            # 解析 entity_type 和 description（假设格式：name, a TYPE, DESC）
            src_parts = src_text.split(",", 2)
            tgt_parts = tgt_text.split(",", 2)

            src_type = src_parts[1].strip() if len(src_parts) > 1 else ""
            src_desc = src_parts[2].strip() if len(src_parts) > 2 else ""
            tgt_type = tgt_parts[1].strip() if len(tgt_parts) > 1 else ""
            tgt_desc = tgt_parts[2].strip() if len(tgt_parts) > 2 else ""

            src_sentence = f"The entity {src} is {src_type} with the description({src_desc})"
            tgt_sentence = f"The entity {tgt} is {tgt_type} with the description({tgt_desc})"
            edge_sentence = f" through edge ({rel}) to connect {src} and {tgt}."

            # 拼接为一个段落
            parts.append(src_sentence + edge_sentence + tgt_sentence)

        return " and ".join(parts)

    def path_to_natural_sentence(path, edge_list, entity_info_map):
        if not path:
            return ""
        text_parts = []
        start = path[0]
        start_info = entity_info_map.get(start, start)
        text_parts.append(f"{start_info}")
        for i in range(len(edge_list)):
            rel = edge_list[i]
            tgt = path[i + 1]
            if i == 0:
                text_parts.append(f"had a relation '{rel}' with {tgt}")
            else:
                text_parts.append(f", who in turn had a '{rel}' relation with {tgt}")
        last_node = path[-1]
        if last_node in entity_info_map:
            text_parts.append(f". {last_node} is {entity_info_map[last_node].split(',', 1)[1]}.")
        return " ".join(text_parts)

    selected_nodes = get_target_nodes(G, question_embedding)
    for src in source_nodes:
        for tgt in selected_nodes:
            if src == tgt:
                continue
            try:
                for path in nx.all_simple_paths(G, source=src, target=tgt, cutoff=max_hops):
                    if len(path) < 2:
                        continue

                    edge_list = []
                    for i in range(len(path) - 1):
                        edge_data = G.get_edge_data(path[i], path[i+1])
                        if edge_data is None:
                            break
                        if isinstance(edge_data, dict):
                            edge_info = list(edge_data.values())[0]
                        else:
                            edge_info = edge_data
                        edge_list.append(edge_info)
                    else:
                        # 查询实体信息
                        entity_info_map = {}
                        for node in path:
                            if node not in added_entities:
                                node_data = await knowledge_graph_inst.get_node(node)
                                if node_data:
                                    entity_info = f"{node}, a {node_data['entity_type']}, {node_data['description']}"
                                else:
                                    entity_info = node
                                added_entities[node] = entity_info
                            entity_info_map[node] = added_entities[node]

                        sentence = path_to_natural_sentence3(path, edge_list, entity_info_map)
                        # sentence = await format_kg_path_description(path, knowledge_graph_inst)
                        all_results.append((path, sentence))
            except Exception:
                continue

    path_list = [i[-1] for i in all_results]
    path_embedding = bge_model.encode(path_list, show_progress_bar=False)
    path_question_similarity = bge_model.similarity(path_embedding, question_embedding[1:]).cpu().numpy()
    selected_path = select_top_paths_per_subq_no_repeat(path_question_similarity, path_list, path_count)
    selected_path = [item for sublist in selected_path.values() for item in sublist]
    all_results = []
    for i, _, path_text in selected_path:
        all_results.append([np.dot(path_question_similarity[i], question_importance), path_text])
    all_results = sorted(all_results, key=lambda x: x[0], reverse=True)
    return all_results

async def _find_most_related_edges_from_entities_pathrag(subgraph, use_model_func, question, source_nodes, knowledge_graph_inst):
    question_embedding = await get_question_embedding(use_model_func, question)
    target_nodes = get_target_nodes(subgraph, question_embedding)

    result, path_stats, one_hop_paths, two_hop_paths, three_hop_paths = await find_paths_and_edges_with_stats(subgraph,
                                                                                                              source_nodes,
                                                                                                              target_nodes)

    threshold = 0.3
    alpha = 0.8
    all_results = []

    for node1 in source_nodes:
        for node2 in target_nodes:
            if node1 != node2:
                if (node1, node2) in result:
                    sub_G = nx.Graph()
                    paths = result[(node1, node2)]['paths']
                    edges = result[(node1, node2)]['edges']
                    sub_G.add_edges_from(edges)
                    results = bfs_weighted_paths(subgraph, paths, node1, node2, threshold, alpha)
                    all_results += results
    all_results = sorted(all_results, key=lambda x: x[1], reverse=True)
    seen = set()
    result_edge = []
    for edge, weight in all_results:
        sorted_edge = tuple(sorted(edge))
        if sorted_edge not in seen:
            seen.add(sorted_edge)
            result_edge.append((edge, weight))

    length_1 = int(len(one_hop_paths) / 2)
    length_2 = int(len(two_hop_paths) / 2)
    length_3 = int(len(three_hop_paths) / 2)
    results = []
    if one_hop_paths != []:
        results = one_hop_paths[0:length_1]
    if two_hop_paths != []:
        results = results + two_hop_paths[0:length_2]
    if three_hop_paths != []:
        results = results + three_hop_paths[0:length_3]

    length = len(results)
    total_edges = 15
    if length < total_edges:
        total_edges = length
    sort_result = []
    if result_edge:
        if len(result_edge) > total_edges:
            sort_result = result_edge[0:total_edges]
        else:
            sort_result = result_edge
    final_result = []
    for edge, weight in sort_result:
        final_result.append(edge)

    relationship = []

    for path in final_result:
        if len(path) == 4:
            s_name, b1_name, b2_name, t_name = path[0], path[1], path[2], path[3]
            edge0 = await knowledge_graph_inst.get_edge(path[0], path[1]) or await knowledge_graph_inst.get_edge(
                path[1], path[0])
            edge1 = await knowledge_graph_inst.get_edge(path[1], path[2]) or await knowledge_graph_inst.get_edge(
                path[2], path[1])
            edge2 = await knowledge_graph_inst.get_edge(path[2], path[3]) or await knowledge_graph_inst.get_edge(
                path[3], path[2])
            if edge0 == None or edge1 == None or edge2 == None:
                print(path, "边丢失")
                if edge0 == None:
                    print("edge0丢失")
                if edge1 == None:
                    print("edge1丢失")
                if edge2 == None:
                    print("edge2丢失")
                continue
            e1 = "through edge (" + edge0["keywords"] + ") to connect to " + s_name + " and " + b1_name + "."
            e2 = "through edge (" + edge1["keywords"] + ") to connect to " + b1_name + " and " + b2_name + "."
            e3 = "through edge (" + edge2["keywords"] + ") to connect to " + b2_name + " and " + t_name + "."
            s = await knowledge_graph_inst.get_node(s_name)
            s = "The entity " + s_name + " is a " + s["entity_type"] + " with the description(" + s["description"] + ")"
            b1 = await knowledge_graph_inst.get_node(b1_name)
            b1 = "The entity " + b1_name + " is a " + b1["entity_type"] + " with the description(" + b1[
                "description"] + ")"
            b2 = await knowledge_graph_inst.get_node(b2_name)
            b2 = "The entity " + b2_name + " is a " + b2["entity_type"] + " with the description(" + b2[
                "description"] + ")"
            t = await knowledge_graph_inst.get_node(t_name)
            t = "The entity " + t_name + " is a " + t["entity_type"] + " with the description(" + t["description"] + ")"
            relationship.append([s + e1 + b1 + "and" + b1 + e2 + b2 + "and" + b2 + e3 + t])
        elif len(path) == 3:
            s_name, b_name, t_name = path[0], path[1], path[2]
            edge0 = await knowledge_graph_inst.get_edge(path[0], path[1]) or await knowledge_graph_inst.get_edge(
                path[1], path[0])
            edge1 = await knowledge_graph_inst.get_edge(path[1], path[2]) or await knowledge_graph_inst.get_edge(
                path[2], path[1])
            if edge0 == None or edge1 == None:
                print(path, "边丢失")
                continue
            e1 = "through edge(" + edge0["keywords"] + ") to connect to " + s_name + " and " + b_name + "."
            e2 = "through edge(" + edge1["keywords"] + ") to connect to " + b_name + " and " + t_name + "."
            s = await knowledge_graph_inst.get_node(s_name)
            s = "The entity " + s_name + " is a " + s["entity_type"] + " with the description(" + s["description"] + ")"
            b = await knowledge_graph_inst.get_node(b_name)
            b = "The entity " + b_name + " is a " + b["entity_type"] + " with the description(" + b["description"] + ")"
            t = await knowledge_graph_inst.get_node(t_name)
            t = "The entity " + t_name + " is a " + t["entity_type"] + " with the description(" + t["description"] + ")"
            relationship.append([s + e1 + b + "and" + b + e2 + t])
        elif len(path) == 2:
            s_name, t_name = path[0], path[1]
            edge0 = await knowledge_graph_inst.get_edge(path[0], path[1]) or await knowledge_graph_inst.get_edge(
                path[1], path[0])
            if edge0 == None:
                print(path, "边丢失")
                continue
            e = "through edge(" + edge0["keywords"] + ") to connect to " + s_name + " and " + t_name + "."
            s = await knowledge_graph_inst.get_node(s_name)
            s = "The entity " + s_name + " is a " + s["entity_type"] + " with the description(" + s["description"] + ")"
            t = await knowledge_graph_inst.get_node(t_name)
            t = "The entity " + t_name + " is a " + t["entity_type"] + " with the description(" + t["description"] + ")"
            relationship.append([s + e + t])
    return relationship

async def _find_most_related_edges_from_entities3(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
    question,
    use_model_func,
    embedding_func,
    total_edges=15
):
    G = nx.Graph()
    edges = await knowledge_graph_inst.edges()
    nodes = await knowledge_graph_inst.nodes()

    edge_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(s, t) for s,t in list(edges)]
    )

    edge_datas = [
        {"src_id": k[0], "tgt_id": k[1], **v}
        for k, v in zip(list(edges), edge_datas)
        if v is not None
    ]

    for e in edge_datas:
        G.add_edge(e['src_id'], e['tgt_id'], relation=e['keywords'])
    G.add_nodes_from(nodes)
    source_nodes = [dp["entity_name"] for dp in node_datas]

    subgraph = extract_k_hop_subgraph(G, source_nodes)
    # 移除孤立节点（degree == 0）
    # isolated_nodes = list(nx.isolates(subgraph))
    # subgraph.remove_nodes_from(isolated_nodes)

    all_results = await extract_and_rank_natural_paths(subgraph, source_nodes, question, knowledge_graph_inst, use_model_func, embedding_func, top_k_paths=total_edges)
    relationship = [i[-1] for i in all_results]

    # relationship = await _find_most_related_edges_from_entities_pathrag(subgraph, use_model_func, question, source_nodes, knowledge_graph_inst)

    relationship = truncate_list_by_token_size(
          relationship, 
          key=lambda x: x[0],
          max_token_size=query_param.max_token_for_local_context,
    )

    reversed_relationship = relationship[::-1]
    return reversed_relationship