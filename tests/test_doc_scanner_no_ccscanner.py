"""
测试文档扫描流程（不调用 ccscanner）

本测试专注于验证：
1. 文档文件的正则匹配和提取
2. 文档内容读取和URL/文件引用提取
3. URL 抓取和摘要生成（可选）
4. 文档内容的过滤和重组
5. 构建信息的提取（基于文档内容）

不包含：
- ccscanner 调用和依赖扫描
"""
import asyncio
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# 添加项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from repo_parse.doc_scanner_parse import (
    read_document_with_extraction,
    read_multiple_documents_parallel,
    fetch_url_with_retry,
    smart_summarize_urls_parallel,
    filter_and_reorganize_docs,
    process_referenced_files,
    EXTRACT_BUILD_INFO_SYSTEM
)
from repo_parse.file_matcher import (
    extract_build_related_files,
    filter_by_keyword_content,
    post_process_matched_files
)
from repo_parse.utils import parse_level2_directory, regex_get_urls
from llm.ask_llm import ask_ds


class MockBuildTarget:
    """模拟 BuildAutomation 对象用于测试"""
    def __init__(self, compile_path, output_dir=None):
        self.compile_path = compile_path
        self.sysuse = "trixie"
        
        # 为每个仓库创建独立的输出目录
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            repo_name = Path(compile_path).name
            self.agent_log_dir = f"/tmp/test_doc_scanner_no_ccscanner/{repo_name}_{timestamp}"
        else:
            self.agent_log_dir = output_dir
        
        os.makedirs(self.agent_log_dir, exist_ok=True)
        
        # 属性用于存储结果
        self.docs = ""
        self.files = []
        self.urlsummaries = {}
        self.summary = ""
    
    def save_result(self, filename, content):
        """保存结果到文件"""
        try:
            filepath = os.path.join(self.agent_log_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  ✓ Saved: {filename}")
            return (True, "")
        except Exception as e:
            print(f"  ✗ Failed to save {filename}: {e}")
            return (False, str(e))


async def test_file_extraction(repo_path):
    """
    测试使用正则模式提取构建相关文件
    """
    print(f"\n{'='*80}")
    print("Test 1: Build-related File Extraction (Regex)")
    print('='*80)
    
    # 1. 使用正则匹配器提取所有构建相关文档和 CI/CD 文件
    print(f"→ Scanning repository: {repo_path}")
    matched_files = extract_build_related_files(repo_path, max_depth=10)
    print(f"  Found {len(matched_files)} files matching regex patterns")
    
    # 2. 根据内容关键字进一步过滤
    print(f"\n→ Filtering by keyword content...")
    keyword_filtered_files = filter_by_keyword_content(
        matched_files,
        keywords=["install", "require", "apt-get", "apt install", "dependenc", 
                 "prerequisites", "compilation", "build", "cmake", "configure"]
    )
    print(f"  {len(keyword_filtered_files)} files contain relevant keywords")
    
    # 3. 后处理：优先级排序和数量限制
    print(f"\n→ Post-processing and prioritizing files...")
    build_related_files = post_process_matched_files(keyword_filtered_files, max_files=20)
    print(f"  Selected {len(build_related_files)} high-priority files")
    
    # 4. 显示提取的文件示例
    if build_related_files:
        print(f"\n→ Sample extracted files:")
        for i, f in enumerate(build_related_files[:10], 1):
            rel_path = os.path.relpath(f, repo_path)
            file_size = os.path.getsize(f)
            print(f"  {i:2}. {rel_path:<50} ({file_size:>8} bytes)")
        if len(build_related_files) > 10:
            print(f"     ... and {len(build_related_files) - 10} more files")
    
    print(f"\n✅ File extraction test completed\n")
    return build_related_files


async def test_document_reading(file_list, repo_path):
    """
    测试并行读取多个文档并提取 URL 和文件引用
    """
    print(f"\n{'='*80}")
    print("Test 2: Document Reading and URL/File Extraction")
    print('='*80)
    
    if not file_list:
        print("⚠ No files to read")
        return "", [], []
    
    print(f"→ Reading {len(file_list)} documents in parallel...")
    
    # 并行读取所有文档
    docs_content, extracted_urls, extracted_files = await read_multiple_documents_parallel(
        file_list[:10],  # 限制文件数量以加快测试
        repo_path
    )
    
    print(f"\n→ Results:")
    print(f"  - Total content length: {len(docs_content)} chars")
    print(f"  - URLs found: {len(extracted_urls)}")
    print(f"  - File references found: {len(extracted_files)}")
    
    # 显示提取的 URL 示例
    if extracted_urls:
        print(f"\n→ Sample extracted URLs:")
        for i, url in enumerate(extracted_urls[:5], 1):
            print(f"  {i}. {url}")
        if len(extracted_urls) > 5:
            print(f"     ... and {len(extracted_urls) - 5} more URLs")
    
    # 显示提取的文件引用示例
    if extracted_files:
        print(f"\n→ Sample file references:")
        for i, file_ref in enumerate(extracted_files[:5], 1):
            print(f"  {i}. {file_ref}")
        if len(extracted_files) > 5:
            print(f"     ... and {len(extracted_files) - 5} more files")
    
    print(f"\n✅ Document reading test completed\n")
    return docs_content, extracted_urls, extracted_files


async def test_url_summarization(urls, repo_path, enable_llm=False):
    """
    测试 URL 抓取和摘要（可选是否启用 LLM）
    """
    print(f"\n{'='*80}")
    print("Test 3: URL Fetching and Summarization (Optional)")
    print('='*80)
    
    if not urls:
        print("⚠ No URLs to process")
        return {}
    
    if not enable_llm:
        print("⚠ LLM summarization disabled (set enable_llm=True to enable)")
        print(f"  URLs available: {len(urls)}")
        return {}
    
    print(f"→ Processing up to 5 URLs with retry mechanism...")
    
    # 并行抓取和总结 URL
    url_summaries = await smart_summarize_urls_parallel(urls[:5], repo_path)
    
    print(f"\n→ Results:")
    print(f"  - Successfully summarized: {len(url_summaries)} URLs")
    
    # 显示摘要示例
    if url_summaries:
        print(f"\n→ Sample URL summaries:")
        for i, (url, summary) in enumerate(list(url_summaries.items())[:3], 1):
            summary_preview = summary[:150] + "..." if len(summary) > 150 else summary
            print(f"  {i}. {url}")
            print(f"     {summary_preview}")
    
    print(f"\n✅ URL summarization test completed\n")
    return url_summaries


async def test_document_filtering(docs_content):
    """
    测试文档内容的过滤和重组
    """
    print(f"\n{'='*80}")
    print("Test 4: Document Filtering and Reorganization")
    print('='*80)
    
    if not docs_content or len(docs_content) < 100:
        print("⚠ Insufficient document content to filter")
        return docs_content
    
    print(f"→ Original document size: {len(docs_content)} chars")
    
    # 过滤和重组文档
    filtered_content = await filter_and_reorganize_docs(
        docs_content, 
        {},  # 不传入 URL 摘要
        []   # 不传入依赖信息
    )
    
    print(f"→ Filtered document size: {len(filtered_content)} chars")
    print(f"→ Reduction: {len(docs_content) - len(filtered_content)} chars ({100 - len(filtered_content)*100//len(docs_content) if docs_content else 0}%)")
    
    print(f"\n✅ Document filtering test completed\n")
    return filtered_content


async def test_build_info_extraction(docs_content, enable_llm=False):
    """
    测试从文档中提取构建信息
    """
    print(f"\n{'='*80}")
    print("Test 5: Build Information Extraction")
    print('='*80)
    
    if not docs_content:
        print("⚠ No document content available")
        return "None"
    
    if not enable_llm:
        print("⚠ LLM extraction disabled (set enable_llm=True to enable)")
        print(f"  Document content available: {len(docs_content)} chars")
        return "LLM disabled - skipped"
    
    print(f"→ Extracting build information with LLM...")
    
    try:
        # 限制输入长度
        doc_input = docs_content[:50000]
        
        build_info_prompt = f"""
Documentation content:
{doc_input}

Please extract build information for X64 Linux (Debian).
"""
        
        build_info = ask_ds(
            system_prompt=EXTRACT_BUILD_INFO_SYSTEM,
            user_prompt=build_info_prompt
        )
        
        # 如果返回的是列表，取第一个元素
        if isinstance(build_info, list):
            build_info = build_info[0] if build_info else "None"
        
        print(f"\n→ Build information extracted:")
        preview = build_info[:300] + "..." if len(build_info) > 300 else build_info
        print(f"  {preview}")
        
        print(f"\n✅ Build info extraction test completed\n")
        return build_info
        
    except Exception as e:
        print(f"✗ Error extracting build info: {e}")
        return "Error during extraction"


async def test_referenced_files(file_refs, repo_path, visited_files):
    """
    测试处理引用的文件
    """
    print(f"\n{'='*80}")
    print("Test 6: Referenced File Processing")
    print('='*80)
    
    if not file_refs:
        print("⚠ No file references to process")
        return "", [], []
    
    print(f"→ Processing {len(file_refs)} referenced files...")
    
    # 转换相对路径为绝对路径
    absolute_file_refs = [
        os.path.join(repo_path, f) if not os.path.isabs(f) else f
        for f in file_refs
    ]
    
    # 处理引用的文件
    new_content, new_urls, new_files = await process_referenced_files(
        absolute_file_refs[:5],  # 限制数量
        repo_path,
        visited_files
    )
    
    print(f"\n→ Results:")
    print(f"  - Additional content: {len(new_content)} chars")
    print(f"  - New URLs found: {len(new_urls)}")
    print(f"  - New file references: {len(new_files)}")
    
    print(f"\n✅ Referenced file processing test completed\n")
    return new_content, new_urls, new_files


async def test_full_workflow_no_ccscanner(repo_path, enable_llm=False, output_dir=None):
    """
    完整的测试工作流（不调用 ccscanner）
    
    Args:
        repo_path: 仓库路径
        enable_llm: 是否启用 LLM 相关功能（URL 摘要、构建信息提取）
        output_dir: 输出目录（可选）
    
    Returns:
        dict: 测试结果摘要
    """
    print(f"\n{'#'*80}")
    print(f"# Full Workflow Test (Without CCScanner)")
    print(f"# Repository: {repo_path}")
    print(f"# LLM Enabled: {enable_llm}")
    print(f"{'#'*80}\n")
    
    if not os.path.exists(repo_path):
        print(f"❌ Repository not found: {repo_path}")
        return None
    
    # 创建模拟对象
    mock_target = MockBuildTarget(repo_path, output_dir)
    print(f"→ Output directory: {mock_target.agent_log_dir}\n")
    
    try:
        # Test 1: 提取构建相关文件
        build_files = await test_file_extraction(repo_path)
        
        # Test 2: 读取文档并提取 URL/文件引用
        docs_content, urls, file_refs = await test_document_reading(build_files, repo_path)
        
        # Test 3: URL 摘要（可选）
        url_summaries = await test_url_summarization(urls, repo_path, enable_llm)
        
        # Test 4: 文档过滤
        filtered_docs = await test_document_filtering(docs_content)
        
        # Test 5: 提取构建信息（可选）
        build_info = await test_build_info_extraction(filtered_docs, enable_llm)
        
        # Test 6: 处理引用的文件（递归）
        visited_files = set(build_files)
        additional_content, new_urls, new_files = await test_referenced_files(
            file_refs, repo_path, visited_files
        )
        
        # 保存结果
        print(f"\n{'='*80}")
        print("Saving Results")
        print('='*80)
        
        final_docs = filtered_docs
        if additional_content:
            final_docs += "\n\n" + additional_content
        
        mock_target.docs = final_docs[:10000] if final_docs else "No documentation available"
        mock_target.files = list(visited_files)
        mock_target.urlsummaries = {
            "url_summaries": url_summaries,
            "processed_urls": urls,
            "processed_files": list(visited_files)
        }
        
        # 保存到文件
        mock_target.save_result("doc_build_info.txt", build_info)
        mock_target.save_result("url_summaries.json", json.dumps(url_summaries, indent=2, ensure_ascii=False))
        mock_target.save_result("processed_files.json", json.dumps(list(visited_files), indent=2))
        mock_target.save_result("full_docs.txt", final_docs)
        
        # 创建测试摘要
        summary = {
            "repo_path": repo_path,
            "repo_name": Path(repo_path).name,
            "timestamp": datetime.now().isoformat(),
            "llm_enabled": enable_llm,
            "files_extracted": len(build_files),
            "files_processed": len(visited_files),
            "urls_found": len(urls),
            "url_summaries_count": len(url_summaries),
            "docs_length": len(final_docs),
            "build_info": build_info if enable_llm else "LLM disabled",
            "output_dir": mock_target.agent_log_dir,
            "tests_completed": [
                "file_extraction",
                "document_reading",
                "url_summarization" if enable_llm else "url_summarization_skipped",
                "document_filtering",
                "build_info_extraction" if enable_llm else "build_info_extraction_skipped",
                "referenced_files"
            ]
        }
        
        mock_target.save_result("test_summary.json", json.dumps(summary, indent=2, ensure_ascii=False))
        
        # 打印最终摘要
        print(f"\n{'#'*80}")
        print("# Test Summary")
        print(f"{'#'*80}")
        print(f"  Repository: {Path(repo_path).name}")
        print(f"  Files extracted: {len(build_files)}")
        print(f"  Files processed: {len(visited_files)}")
        print(f"  URLs found: {len(urls)}")
        print(f"  URL summaries: {len(url_summaries)}")
        print(f"  Final docs length: {len(final_docs)} chars")
        print(f"  Output dir: {mock_target.agent_log_dir}")
        print(f"{'#'*80}\n")
        
        print("✅ All tests completed successfully!\n")
        
        return summary
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return None


async def main():
    """主测试函数"""
    # 配置测试
    test_repos = [
        Path(__file__).resolve().parent.parent,  # 本项目
        # 添加更多测试仓库...
    ]
    
    # 是否启用 LLM 功能（URL摘要、构建信息提取）
    # 设置为 False 可以快速测试基础功能
    enable_llm = True
    
    print("=" * 80)
    print("Document Scanner Test Suite (Without CCScanner)")
    print("=" * 80)
    print(f"Test repositories: {len(test_repos)}")
    print(f"LLM features: {'Enabled' if enable_llm else 'Disabled'}")
    print("=" * 80)
    
    results = []
    for repo in test_repos:
        result = await test_full_workflow_no_ccscanner(
            repo, 
            enable_llm=enable_llm
        )
        if result:
            results.append(result)
    
    # 汇总所有测试结果
    if results:
        print(f"\n{'='*80}")
        print(f"All Tests Summary ({len(results)} repositories)")
        print('='*80)
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['repo_name']}")
            print(f"   - Files: {result['files_processed']}")
            print(f"   - URLs: {result['urls_found']}")
            print(f"   - Output: {result['output_dir']}")
        print('='*80)


if __name__ == "__main__":
    asyncio.run(main())
