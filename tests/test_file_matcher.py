#!/usr/bin/env python3
"""
测试新的正则文件匹配功能
"""
import sys
import os
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from repo_parse.file_matcher import (
    extract_build_related_files,
    filter_by_keyword_content,
    post_process_matched_files,
    build_doc_matcher,
    build_cicd_matcher
)


def test_file_extraction(repo_path: str):
    """测试文件提取功能"""
    print("=" * 80)
    print(f"测试仓库: {repo_path}")
    print("=" * 80)
    
    if not os.path.exists(repo_path):
        print(f"❌ 仓库不存在: {repo_path}")
        return
    
    # 1. 提取所有匹配的文件
    print("\n📁 步骤 1: 使用正则匹配器扫描文件...")
    matched_files = extract_build_related_files(repo_path, max_depth=10)
    print(f"   找到 {len(matched_files)} 个匹配的文件")
    
    if matched_files:
        print("\n   示例匹配文件:")
        for i, f in enumerate(matched_files[:10], 1):
            rel_path = os.path.relpath(f, repo_path)
            print(f"   {i:2}. {rel_path}")
        if len(matched_files) > 10:
            print(f"   ... 还有 {len(matched_files) - 10} 个文件")
    
    # 2. 按关键字内容过滤
    print("\n🔍 步骤 2: 根据内容关键字过滤...")
    filtered_files = filter_by_keyword_content(matched_files)
    print(f"   {len(filtered_files)} 个文件包含相关关键字")
    
    # 3. 后处理和优先级排序
    print("\n⚡ 步骤 3: 优先级排序和数量限制...")
    final_files = post_process_matched_files(filtered_files, max_files=15)
    print(f"   选择了 {len(final_files)} 个高优先级文件\n")
    
    if final_files:
        print("   最终选择的文件（按优先级）:")
        print("   " + "-" * 76)
        for i, f in enumerate(final_files, 1):
            rel_path = os.path.relpath(f, repo_path)
            file_size = os.path.getsize(f)
            size_kb = file_size / 1024
            print(f"   {i:2}. {rel_path:<60} ({size_kb:.1f} KB)")
    
    # 4. 分类统计
    print("\n📊 步骤 4: 文件类型统计...")
    doc_matcher = build_doc_matcher()
    cicd_matcher = build_cicd_matcher()
    
    doc_count = sum(1 for f in final_files if doc_matcher(f))
    cicd_count = sum(1 for f in final_files if cicd_matcher(f))
    
    print(f"   文档文件: {doc_count}")
    print(f"   CI/CD文件: {cicd_count}")
    
    # 5. 按文件名类型分组
    print("\n📋 步骤 5: 按文件名分组...")
    file_groups = {
        "README": [],
        "INSTALL": [],
        "BUILDING": [],
        "DEPEND/REQUIRE": [],
        "CI/CD": [],
        "其他": []
    }
    
    for f in final_files:
        name = os.path.basename(f).lower()
        if 'readme' in name:
            file_groups["README"].append(f)
        elif 'install' in name:
            file_groups["INSTALL"].append(f)
        elif 'build' in name:
            file_groups["BUILDING"].append(f)
        elif any(kw in name for kw in ['depend', 'require']):
            file_groups["DEPEND/REQUIRE"].append(f)
        elif cicd_matcher(f):
            file_groups["CI/CD"].append(f)
        else:
            file_groups["其他"].append(f)
    
    for group_name, files in file_groups.items():
        if files:
            print(f"   {group_name}: {len(files)} 个文件")
    
    print("\n" + "=" * 80)
    print("✅ 测试完成")
    print("=" * 80)


def main():
    """主测试函数"""
    # 测试多个仓库
    test_repos = [
        Path(__file__).resolve().parent.parent,  # 当前项目
        "<test_repo_path>/admesh_admesh",
        "<test_repo_path>/01org_isa-l",
    ]
    
    # 如果命令行提供了路径，使用命令行参数
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        # 特殊模式：批量测试模式
        if arg == "--batch":
            if len(sys.argv) > 2:
                base_dir = sys.argv[2]
                output_json = sys.argv[3] if len(sys.argv) > 3 else None
                batch_test_directories(base_dir, output_json)
            else:
                print("用法: python test_file_matcher.py --batch <base_directory> [output_json]")
                print("示例: python test_file_matcher.py --batch /path/to/collects results.json")
            return
        else:
            test_repos = [arg]
    
    for repo in test_repos:
        if os.path.exists(repo):
            test_file_extraction(repo)
            print("\n\n")
            break  # 只测试第一个存在的仓库
        else:
            print(f"⚠️  仓库不存在，跳过: {repo}")


def batch_test_directories(base_dir: str, output_json: str = None):
    """
    批量测试指定目录下第二层的所有文件夹
    
    Args:
        base_dir: 基础目录路径
        output_json: 输出 JSON 文件路径（可选）
    """
    print("=" * 80)
    print(f"批量测试模式")
    print(f"基础目录: {base_dir}")
    print("=" * 80)
    
    if not os.path.exists(base_dir):
        print(f"❌ 目录不存在: {base_dir}")
        return
    
    # 使用 find 命令获取第二层目录
    print("\n📂 扫描第二层目录...")
    try:
        result = subprocess.run(
            ["find", base_dir, "-maxdepth", "2", "-mindepth", "2", "-type", "d"],
            capture_output=True,
            text=True,
            check=True
        )
        directories = [d.strip() for d in result.stdout.split('\n') if d.strip()]
    except subprocess.CalledProcessError as e:
        print(f"❌ 扫描失败: {e}")
        return
    
    print(f"   找到 {len(directories)} 个目录\n")
    
    if not directories:
        print("⚠️  没有找到任何目录")
        return
    
    # 准备结果数据结构
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "base_directory": base_dir,
        "total_directories": len(directories),
        "results": {}
    }
    
    # 统计信息
    stats = {
        "processed": 0,
        "failed": 0,
        "total_matched_files": 0,
        "total_filtered_files": 0,
        "total_final_files": 0
    }
    
    # 批量处理每个目录
    for i, repo_path in enumerate(directories, 1):
        repo_name = "/".join(repo_path.split("/")[-2:])  # 获取最后两层路径作为名称
        
        print(f"[{i}/{len(directories)}] 测试: {repo_name}")
        
        try:
            # 1. 提取所有匹配的文件
            matched_files = extract_build_related_files(repo_path, max_depth=10)
            
            # 2. 按关键字内容过滤
            filtered_files = filter_by_keyword_content(matched_files)
            
            # 3. 后处理和优先级排序
            final_files = post_process_matched_files(filtered_files, max_files=30)
            
            # 转换为相对路径
            matched_files_rel = [os.path.relpath(f, repo_path) for f in matched_files]
            filtered_files_rel = [os.path.relpath(f, repo_path) for f in filtered_files]
            final_files_rel = [os.path.relpath(f, repo_path) for f in final_files]
            
            # 保存结果
            all_results["results"][repo_name] = {
                "repo_path": repo_path,
                "matched_files": matched_files_rel,
                "filtered_files": filtered_files_rel,
                "final_files": final_files_rel,
                "counts": {
                    "matched": len(matched_files),
                    "filtered": len(filtered_files),
                    "final": len(final_files)
                }
            }
            
            # 更新统计
            stats["processed"] += 1
            stats["total_matched_files"] += len(matched_files)
            stats["total_filtered_files"] += len(filtered_files)
            stats["total_final_files"] += len(final_files)
            
            print(f"   ✓ 匹配: {len(matched_files)} → 过滤: {len(filtered_files)} → 最终: {len(final_files)}")
            
        except Exception as e:
            print(f"   ✗ 失败: {e}")
            stats["failed"] += 1
            all_results["results"][repo_name] = {
                "repo_path": repo_path,
                "error": str(e)
            }
    
    # 添加统计信息
    all_results["statistics"] = stats
    
    # 保存到 JSON 文件
    if output_json is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_json = f"file_matcher_results_{timestamp}.json"
    
    print(f"\n💾 保存结果到: {output_json}")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 打印汇总统计
    print("\n" + "=" * 80)
    print("📊 批量测试汇总")
    print("=" * 80)
    print(f"总目录数:       {len(directories)}")
    print(f"成功处理:       {stats['processed']}")
    print(f"失败:           {stats['failed']}")
    print(f"总匹配文件:     {stats['total_matched_files']}")
    print(f"总过滤文件:     {stats['total_filtered_files']}")
    print(f"总最终文件:     {stats['total_final_files']}")
    print(f"\n平均每个仓库:")
    if stats['processed'] > 0:
        print(f"  匹配文件:     {stats['total_matched_files'] / stats['processed']:.1f}")
        print(f"  过滤文件:     {stats['total_filtered_files'] / stats['processed']:.1f}")
        print(f"  最终文件:     {stats['total_final_files'] / stats['processed']:.1f}")
    print("=" * 80)
    print(f"\n✅ 结果已保存到: {output_json}")


if __name__ == "__main__":
    main()
