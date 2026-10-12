import json

from .utils import parse_level2_directory, regex_get_urls, Fetch_URL_Content, summarize_urls
from .ReadPrompt import GET_RELATED_FILEANDURLS_SYS, GET_RELATED_FILEANDURLS_USR, URL_SUMMARY_SYS, URL_SUMMARY_USR

import sys
sys.path.append("..")
import llm.ask_llm


async def get_relateds(BTarget):
    # 解析前两层目录的内容
    print("compilepath:",BTarget.compile_path)
    compile_path = BTarget.compile_path
    directory_content = await parse_level2_directory(compile_path)
    urls = []
    urls, files = await regex_get_urls(directory_content["files"], compile_path)
    askcontent = {
        "files": directory_content["files"],
        "directories": directory_content["directories"],
        "urls_and_docs": urls+files,
    }
    # LLM返回需要的文件和URL：编译文档、编译链接、配置/编译文件
    llm_related = {}

    for attempt in range(1, 3):
        llm_related = ask_llm.ask_ds(
            GET_RELATED_FILEANDURLS_SYS,
            GET_RELATED_FILEANDURLS_USR.format(
                target_path=compile_path,
                tree=json.dumps(askcontent, ensure_ascii=False, indent=2)
            )
        )
        print(f"尝试第 {attempt} 次，LLM 返回的相关文件和 URL：", llm_related)

        # 如果返回字符串，则尝试转 JSON
        if isinstance(llm_related, list):
            llm_related = llm_related[0]
        if isinstance(llm_related, str):
            if llm_related.startswith("[") and llm_related.endswith("]"):
                llm_related = llm_related[1:-1].strip()
            try:
                llm_related = json.loads(llm_related)
            except json.JSONDecodeError as e:
                print(f"JSON 解析失败（{e}），继续重试")
                continue

        # 检查是否包含所有预期字段
        if all(key in llm_related for key in ("docs", "urls", "files")):
            print("获取成功，退出重试")
            break
    else:
        # 重试结束仍未成功，设置默认值
        print("重试结束，仍未成功，使用默认值填充")
        llm_related = { key: ["not available"] for key in ("docs", "urls", "files") }

    # llm_related = { key: "not available" for key in ("docs", "urls", "files") }
    BTarget.files = llm_related["files"]
    BTarget.docs = llm_related["docs"]
    BTarget.urlsummaries = await summarize_urls(BTarget.compile_path, llm_related["urls"], URL_SUMMARY_SYS, URL_SUMMARY_USR)
    BTarget.save_result(json.dumps(BTarget.urlsummaries, indent=2), "urlsummaries")