"""
测试新的文档扫描并行处理流程
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
    get_input_json,
    read_multiple_documents_parallel,
    get_relateds_from_docs_and_scanner
)


class MockBuildTarget:
    """模拟 BuildAutomation 对象用于测试"""
    def __init__(self, compile_path, output_dir=None):
        self.compile_path = compile_path
        self.sysuse = "trixie"
        
        # 为每个仓库创建独立的输出目录
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            repo_name = Path(compile_path).name
            self.agent_log_dir = f"/tmp/test_agent_logs/{repo_name}_{timestamp}"
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
            return (True, "")
        except Exception as e:
            return (False, str(e))


async def test_document_reading():
    """测试文档读取和 URL/文件提取"""
    print("\n=== Testing document reading and extraction ===")
    test_repo = str(Path(__file__).resolve().parent.parent)
    
    # 测试读取 README.md
    readme = os.path.join(test_repo, "README.md")
    if os.path.exists(readme):
        content, urls, files = await read_multiple_documents_parallel(
            [readme], test_repo
        )
        print(f"Content length: {len(content)}")
        print(f"URLs found: {urls}")
        print(f"Files found: {files}")
    else:
        print("README.md not found")


async def test_full_workflow():
    """测试完整的工作流程"""
    print("\n=== Testing full workflow ===")
    test_repo = str(Path(__file__).resolve().parent.parent)
    
    # 创建模拟对象
    mock_target = MockBuildTarget(test_repo)
    
    # 运行完整流程
    result = await get_relateds_from_docs_and_scanner(mock_target)
    
    print(f"\nResults summary:")
    print(f"  - Dependencies found: {len(result.get('dependencies', []))}")
    print(f"  - URL summaries: {len(result.get('url_summaries', {}))}")
    print(f"  - Build info available: {'Yes' if result.get('build_info') != 'None' else 'No'}")
    print(f"  - Files processed: {len(mock_target.files)}")
    print(f"\nSaved results to: {mock_target.agent_log_dir}")
    
    return result


async def test_single_repo(repo_path, output_base_dir="/tmp/test_results"):
    """
    测试单个仓库的文档和URL读取
    
    Args:
        repo_path: 仓库路径
        output_base_dir: 输出基础目录
        
    Returns:
        dict: 包含所有测试结果的字典
    """
    print(f"\n{'='*80}")
    print(f"Testing repository: {repo_path}")
    print(f"{'='*80}")
    
    if not os.path.exists(repo_path):
        print(f"❌ Repository not found: {repo_path}")
        return None
    
    # 创建输出目录
    repo_name = Path(repo_path).name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base_dir, f"{repo_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建模拟对象
    mock_target = MockBuildTarget(repo_path, output_dir)
    
    try:
        # 运行完整流程
        result = await get_relateds_from_docs_and_scanner(mock_target)
        
        # 保存汇总结果
        summary = {
            "repo_path": repo_path,
            "repo_name": repo_name,
            "timestamp": timestamp,
            "dependencies_count": len(result.get('dependencies', [])),
            "dependencies": list(result.get('dependencies', [])),
            "url_summaries_count": len(result.get('url_summaries', {})),
            "url_summaries": result.get('url_summaries', {}),
            "build_info": result.get('build_info', 'None'),
            "files_processed": mock_target.files,
            "files_count": len(mock_target.files),
            "docs_length": len(mock_target.docs),
            "output_dir": output_dir
        }
        
        # 保存 JSON 格式的结果
        summary_file = os.path.join(output_dir, "summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 打印摘要
        print(f"\n✅ Results for {repo_name}:")
        print(f"  - Dependencies found: {summary['dependencies_count']}")
        
        # 显示依赖详情（带版本号）
        if summary['dependencies']:
            deps = summary['dependencies']
            with_version = len([d for d in deps if d.get('version') != 'unspecified'])
            print(f"    • 带版本号: {with_version}/{len(deps)}")
            print(f"    • 示例依赖:")
            for dep in deps[:5]:
                ver = f"v{dep.get('version')}" if dep.get('version', 'unspecified') != 'unspecified' else ""
                print(f"      - {dep.get('unified_name', dep.get('name', 'unknown')):<20} {ver:<15} ({dep.get('extractor_type', 'unknown')})")
        
        print(f"  - URL summaries: {summary['url_summaries_count']}")
        print(f"  - Files processed: {summary['files_count']}")
        print(f"  - Documents length: {summary['docs_length']} chars")
        print(f"  - Output saved to: {output_dir}")
        
        return summary
        
    except Exception as e:
        print(f"❌ Error processing {repo_path}: {e}")
        import traceback
        traceback.print_exc()
        
        error_info = {
            "repo_path": repo_path,
            "repo_name": repo_name,
            "timestamp": timestamp,
            "error": str(e),
            "output_dir": output_dir
        }
        
        # 保存错误信息
        error_file = os.path.join(output_dir, "error.json")
        with open(error_file, "w", encoding="utf-8") as f:
            json.dump(error_info, f, indent=2, ensure_ascii=False)
        
        return error_info


async def test_multiple_repos(repo_paths, output_base_dir="/tmp/test_results", concurrent=False):
    """
    测试多个仓库的文档和URL读取
    
    Args:
        repo_paths: 仓库路径列表
        output_base_dir: 输出基础目录
        concurrent: 是否并发执行（默认顺序执行）
        
    Returns:
        list: 所有仓库的测试结果列表
    """
    print(f"\n{'='*80}")
    print(f"Testing {len(repo_paths)} repositories")
    print(f"Output directory: {output_base_dir}")
    print(f"Mode: {'Concurrent' if concurrent else 'Sequential'}")
    print(f"{'='*80}")
    
    os.makedirs(output_base_dir, exist_ok=True)
    
    if concurrent:
        # 并发执行
        tasks = [test_single_repo(repo, output_base_dir) for repo in repo_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    else:
        # 顺序执行
        results = []
        for repo in repo_paths:
            result = await test_single_repo(repo, output_base_dir)
            results.append(result)
    
    # 保存总体汇总
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    overall_summary = {
        "timestamp": timestamp,
        "total_repos": len(repo_paths),
        "successful": sum(1 for r in results if r and 'error' not in r),
        "failed": sum(1 for r in results if r and 'error' in r),
        "results": results
    }
    
    summary_file = os.path.join(output_base_dir, f"overall_summary_{timestamp}.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"Overall Summary:")
    print(f"  - Total repositories: {overall_summary['total_repos']}")
    print(f"  - Successful: {overall_summary['successful']}")
    print(f"  - Failed: {overall_summary['failed']}")
    print(f"  - Overall summary saved to: {summary_file}")
    print(f"{'='*80}")
    
    return results


async def main():
    """运行所有测试"""
    print("Starting tests...")
    
    # 从命令行参数读取仓库路径，或使用默认列表
    if len(sys.argv) > 1:
        # 命令行参数可以是单个路径或以逗号分隔的多个路径
        arg = sys.argv[1]
        if os.path.isfile(arg):
            # 如果是文件，从文件中读取路径列表
            with open(arg, 'r') as f:
                repo_paths = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        else:
            # 否则解析为路径列表
            repo_paths = [p.strip() for p in arg.split(',')]
    else:
        # 默认测试仓库列表
        repo_paths = [
            str(Path(__file__).resolve().parent.parent),
            # 添加更多测试仓库路径
        ]
    
    # 输出目录（可以从命令行参数指定）
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/test_results"
    
    # 是否并发执行
    concurrent = "--concurrent" in sys.argv
    
    if len(repo_paths) == 1:
        # 单个仓库
        await test_single_repo(repo_paths[0], output_dir)
    else:
        # 多个仓库
        await test_multiple_repos(repo_paths, output_dir, concurrent)
    
    print("\nAll tests completed!")


if __name__ == "__main__":
    asyncio.run(main())

