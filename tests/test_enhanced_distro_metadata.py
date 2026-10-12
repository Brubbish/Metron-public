from repo_parse.doc_scanner_parse import selected_distro_from_enhanced_result


def test_selected_distro_from_per_project_metadata_exact_path():
    result = {
        "per_project": {
            "/tmp/owner/repo": {"CMakeLists.txt": []},
        },
        "per_project_metadata": {
            "/tmp/owner/repo": {
                "selected_debian_version": "bookworm",
            },
        },
    }

    assert selected_distro_from_enhanced_result(result, "/tmp/owner/repo") == "bookworm"


def test_selected_distro_from_per_project_metadata_basename_fallback():
    result = {
        "per_project": {
            "/other/path/owner_repo": {"CMakeLists.txt": []},
        },
        "per_project_metadata": {
            "/other/path/owner_repo": {
                "selected_debian_version": "bullseye",
            },
        },
    }

    assert selected_distro_from_enhanced_result(result, "/tmp/owner/repo") == "bullseye"


def test_selected_distro_old_single_project_fallback():
    result = {
        "run_config": {
            "debian_version": "trixie",
        },
        "per_project": {
            "/tmp/owner/repo": {"CMakeLists.txt": []},
        },
    }

    assert selected_distro_from_enhanced_result(result, "/tmp/owner/repo") == "trixie"
