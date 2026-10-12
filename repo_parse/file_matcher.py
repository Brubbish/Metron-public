"""
文件匹配器模块 - 基于正则表达式筛选构建相关文档和CI/CD文件
提取和改进自 regex_filter.py
"""
import os
import re
from typing import List, Callable
from pathlib import Path


# 文档类型的正则配置
DOC_FILE_PATTERNS = {
    "path": [],  # 路径正则（暂不使用）
    "name": [
        # 依赖相关文件
        r"^(dep|deps|dependency|dependencies|requirements|requirement|compilation|compile)(\.txt|\.html|\.htm|\.md|\.rst)?$",
        # README, INSTALL 等
        r"^(readme|install)[^/]*(\.txt|\.html|\.htm|\.md|\.rst|\.contrib|\.build|\.any|\.cross|\.unix|\.gnu)?$",
        # 构建配置
        r"^meson_options\.txt$",
        r"^build(\.txt|\.md|\.html|\.htm|\.rst)$",
        # License (可选，帮助识别项目信息)
        r"^(license)(\.txt|\.md|\.html|\.htm|\.rst)?$",
    ]
}

# CI/CD 和构建脚本的正则配置
CICD_FILE_PATTERNS = {
    "path": [
        # Debian packaging
        r"/.*/debian/[^/]*control$",
        # GitHub Actions, GitLab CI 等
        r"/\.?(github/(workflows)?|gitlab|gitlab-ci|cicd|ci|ci_build|circleci)(?:/[^/]+)*[^/]*(docker|install|release|set|unix|linux|debian|ubuntu|centos|dep|ci|auto|codeql|main)[^/]*?(\.sh|\.yml|\.yaml)?$"
    ],
    "name": [
        r"^jenkins",
        r"\.?travis(\.yml|\.sh)$",
        r"\.?appveyor\.yml$",
        r"vcpkg(\.json|\.yml|\.yaml)$",
        r"conanfile(\.txt|\.py)$",
        r"(dep|download|install|compile)[^/]*(\.sh|\.yml|\.yaml)$",
        r"\.cirrus\.yml$",
        r"\.?build\.(sh|yml|yaml)$",
        r"\.gitlab-ci\.yml$",
        r"dockerfile",
        r"fuzz\.sh$"
    ]
}

# 文件扩展名白名单
ALLOWED_EXTENSIONS = {
    'md', 'doc', 'txt', 'html', 'htm', 'rst', 
    'linux', 'unix', 'gnu', 'any', 'cross', 
    'sh', 'yml', 'yaml', 'dox', 'py', 'json', 
    'docker', 'dockerfile', ''
}

# 需要排除的路径关键字
EXCLUDED_PATH_KEYWORDS = [
    '/third', '/3rd',  # 第三方库目录
    '/win32/', '/win64/', '/win/', 'windows',  # Windows
    'macos', '/mac/', 'osx', 'ios', 'apple', 'darwin',  # macOS/iOS
]

# 需要排除的文件名关键字
EXCLUDED_NAME_KEYWORDS = ['mac', 'win', 'darwin', 'osx']

DEFAULT_CONTENT_KEYWORDS = [
    # Build actions and phases
    "build", "building", "built", "compile", "compiling", "compilation",
    "configure", "configuration", "setup", "bootstrap",

    # Build systems and helper tools
    "cmake", "meson", "ninja", "make", "autoconf", "automake", "autotools",
    "pkg-config", "pkgconf", "qmake", "scons", "bazel",

    # Dependency descriptions
    "depend", "dependency", "dependencies", "require", "required",
    "requirements", "prerequisite", "prerequisites", "package", "packages",
    "library", "libraries",

    # Package-manager commands commonly found in docs and CI
    "apt", "apt-get", "apt install", "apt-get install", "apt build-dep",
    "build-dep", "dnf install", "yum install", "apk add", "pacman -s",
    "zypper install",

    # Linux/Debian build context
    "debian", "ubuntu", "linux", "toolchain", "compiler", "gcc", "g++",
    "clang",
]


def build_doc_matcher(path_regex: List[str] = None, name_regex: List[str] = None) -> Callable[[str], bool]:
    """
    构建文档文件匹配器
    
    Args:
        path_regex: 路径正则列表（默认使用 DOC_FILE_PATTERNS["path"]）
        name_regex: 文件名正则列表（默认使用 DOC_FILE_PATTERNS["name"]）
        
    Returns:
        匹配函数，输入文件路径，返回 True 表示匹配
    """
    if path_regex is None:
        path_regex = DOC_FILE_PATTERNS["path"]
    if name_regex is None:
        name_regex = DOC_FILE_PATTERNS["name"]
    
    path_patterns = [re.compile(r, re.IGNORECASE) for r in path_regex]
    name_patterns = [re.compile(r, re.IGNORECASE) for r in name_regex]
    
    def matcher(file_path: str) -> bool:
        """判断文件是否为构建相关文档"""
        # 检查文件是否存在且不是链接
        if os.path.islink(file_path) or not os.path.exists(file_path):
            return False
        
        # 检查文件大小
        try:
            if not os.path.getsize(file_path):
                return False
        except OSError:
            return False
        
        path = os.path.normpath(file_path)
        lower_path = path.lower()
        name = os.path.basename(path)
        
        # 排除 Makefile 和 CMakeLists.txt
        if 'makefile' in name.lower() or name.lower() == 'cmakelists.txt':
            return False
        
        # 检查文件扩展名白名单
        if '.' in name:
            ext = name.split(".")[-1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                return False
        
        # 排除第三方库目录
        path_parts = lower_path.split("/")
        if any(keyword in part for part in path_parts for keyword in ['third', '3rd']):
            return False
        
        # 排除非 Linux 平台文件
        if any(keyword in lower_path for keyword in EXCLUDED_PATH_KEYWORDS):
            return False
        
        if any(keyword in name.lower() for keyword in EXCLUDED_NAME_KEYWORDS):
            return False
        
        # 匹配路径正则或文件名正则
        if any(p.search(path) for p in path_patterns):
            return True
        if any(p.search(name) for p in name_patterns):
            return True
        
        return False
    
    return matcher


def build_cicd_matcher(path_regex: List[str] = None, name_regex: List[str] = None) -> Callable[[str], bool]:
    """
    构建 CI/CD 文件匹配器
    
    Args:
        path_regex: 路径正则列表（默认使用 CICD_FILE_PATTERNS["path"]）
        name_regex: 文件名正则列表（默认使用 CICD_FILE_PATTERNS["name"]）
        
    Returns:
        匹配函数，输入文件路径，返回 True 表示匹配
    """
    if path_regex is None:
        path_regex = CICD_FILE_PATTERNS["path"]
    if name_regex is None:
        name_regex = CICD_FILE_PATTERNS["name"]
    
    path_patterns = [re.compile(r, re.IGNORECASE) for r in path_regex]
    name_patterns = [re.compile(r, re.IGNORECASE) for r in name_regex]
    
    def matcher(file_path: str) -> bool:
        """判断文件是否为 CI/CD 或构建脚本"""
        # 检查文件是否存在且不是链接
        if os.path.islink(file_path) or not os.path.exists(file_path):
            return False
        
        # 检查文件大小
        try:
            if not os.path.getsize(file_path):
                return False
        except OSError:
            return False
        
        path = os.path.normpath(file_path)
        lower_path = path.lower()
        name = os.path.basename(path)
        
        # 排除非 Linux 平台文件
        if any(keyword in lower_path for keyword in EXCLUDED_PATH_KEYWORDS):
            return False
        
        # 匹配路径正则或文件名正则
        if any(p.search(path) for p in path_patterns):
            return True
        if any(p.search(name) for p in name_patterns):
            return True
        
        return False
    
    return matcher


def build_combined_matcher() -> Callable[[str], bool]:
    """
    构建组合匹配器（文档 + CI/CD）
    
    Returns:
        匹配函数
    """
    doc_matcher = build_doc_matcher()
    cicd_matcher = build_cicd_matcher()
    
    def matcher(file_path: str) -> bool:
        """判断文件是否为构建相关文档或 CI/CD 文件"""
        return doc_matcher(file_path) or cicd_matcher(file_path)
    
    return matcher


def extract_build_related_files(repo_path: str, max_depth: int = 10) -> List[str]:
    """
    从仓库中提取所有构建相关的文档和 CI/CD 文件
    
    Args:
        repo_path: 仓库根路径
        max_depth: 最大目录深度（相对于仓库根目录）
        
    Returns:
        匹配的文件路径列表（绝对路径）
    """
    matcher = build_combined_matcher()
    matched_files = []
    
    repo_path = os.path.abspath(repo_path)
    repo_depth = len(Path(repo_path).parts)
    
    for root, dirs, files in os.walk(repo_path):
        # 检查深度
        current_depth = len(Path(root).parts) - repo_depth
        if current_depth > max_depth:
            dirs.clear()  # 不再深入子目录
            continue
        
        # 排除某些目录
        dirs[:] = [d for d in dirs if d not in {
            '.git', '.svn', '.hg', 'node_modules', '__pycache__', 
            'build', 'dist', '.venv', 'venv', 'env', '.env',
            'site-packages', '.mypy_cache', '.pytest_cache', '.tox'
        }]
        
        for filename in files:
            file_path = os.path.join(root, filename)
            if matcher(file_path):
                matched_files.append(file_path)
    
    return sorted(matched_files)


def filter_by_keyword_content(file_paths: List[str], keywords: List[str] = None) -> List[str]:
    """
    根据内容关键字进一步过滤文件
    
    Args:
        file_paths: 文件路径列表
        keywords: 关键字列表（默认为构建相关关键字）
        
    Returns:
        包含关键字的文件路径列表
    """
    if keywords is None:
        keywords = DEFAULT_CONTENT_KEYWORDS
    
    filtered_files = []
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().lower()
                if any(keyword.lower() in content for keyword in keywords):
                    filtered_files.append(file_path)
        except Exception:
            continue
    
    return filtered_files


def post_process_matched_files(file_paths: List[str], max_files: int = 30) -> List[str]:
    """
    后处理匹配的文件：优先级排序和数量限制
    
    Args:
        file_paths: 文件路径列表
        max_files: 最大文件数量
        
    Returns:
        处理后的文件路径列表
    """
    if not file_paths:
        return []
    
    # 按优先级排序
    def get_priority(file_path: str) -> int:
        """计算文件优先级（数字越小优先级越高）"""
        name = os.path.basename(file_path).lower()
        path = file_path.lower()
        depth = len(Path(file_path).parts)
        
        priority = depth * 10  # 基础优先级：深度越小越优先
        
        # README 高优先级
        if 'readme' in name:
            priority -= 100
        
        # INSTALL, BUILDING 高优先级
        if any(kw in name for kw in ['install', 'building', 'compile']):
            priority -= 80
        
        # 依赖文件高优先级
        if any(kw in name for kw in ['depend', 'require']):
            priority -= 60
        
        # CI/CD 文件较高优先级
        if any(kw in path for kw in ['.github', 'gitlab', 'debian']):
            priority -= 40
        
        # 根目录文件优先
        if depth <= 2:
            priority -= 50
        
        return priority
    
    sorted_files = sorted(file_paths, key=get_priority)
    
    # 限制数量
    return sorted_files[:max_files]
