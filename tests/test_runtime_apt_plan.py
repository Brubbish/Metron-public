from core.runtime_apt_plan import normalize_apt_install_command, normalize_apt_packages


def test_legacy_pkgconf_maps_to_pkg_config_and_conflict_family_is_single_choice():
    command = normalize_apt_install_command(
        "apt-get install pkgconf heimdal-dev libkrb5-dev zlib1g-dev",
        sysuse="lenny",
    )

    assert command == "apt-get install -y pkg-config libkrb5-dev zlib1g-dev"


def test_options_are_preserved_when_provider_family_is_reduced():
    command = normalize_apt_install_command(
        "DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends libcurl4-openssl-dev libcurl4-gnutls-dev zlib1g-dev",
        sysuse="bookworm",
    )

    assert "--no-install-recommends" in command
    assert "libcurl4-gnutls-dev" in command
    assert "libcurl4-openssl-dev" not in command
    assert command.endswith("libcurl4-gnutls-dev zlib1g-dev")


def test_dynamic_package_commands_are_left_unchanged():
    command = "apt-get install $(cat deps.txt)"

    assert normalize_apt_install_command(command, sysuse="bookworm") == command


def test_package_list_normalization_is_deduplicated():
    assert normalize_apt_packages(
        ["pkgconf", "pkg-config", "pkgconf", "libkrb5-dev", "heimdal-dev"],
        sysuse="wheezy",
    ) == ["pkg-config", "libkrb5-dev"]
