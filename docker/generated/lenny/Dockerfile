FROM debian/eol:lenny

LABEL org.theauto.repository-profile="e91c36ddc53d6da5"       org.theauto.debian-release="lenny"

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG http_proxy
ARG https_proxy
ARG NO_PROXY
ARG no_proxy

ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}
ENV NO_PROXY=${NO_PROXY}
ENV no_proxy=${no_proxy}

RUN rm -f /etc/apt/sources.list.d/*.sources /etc/apt/sources.list.d/*.list     && rm -rf /var/lib/apt/lists/*     && mkdir -p /var/lib/apt/lists/partial     && mkdir -p /etc/apt/preferences.d     && printf '%s\n' "deb [check-valid-until=no trusted=yes] http://archive.debian.org/debian lenny contrib main non-free" > /etc/apt/sources.list     && printf '%s\n' "Package: *" "Pin: release n=lenny" "Pin-Priority: 1001" "" > /etc/apt/preferences.d/theauto-repositories     && printf '%s\n' '{"architecture":"amd64","archive":true,"base_image":"debian/eol:lenny","base_url":"http://archive.debian.org/debian","components":["main","contrib","non-free"],"profile_hash":"e91c36ddc53d6da5","release":"lenny","repositories":["lenny"],"resolved_profile_hash":"ef1ebc07812ea44a","schema_version":1}' > /etc/theauto-repository-profile.json

RUN /bin/sh -eu <<'EOF'
set -eu

export DEBIAN_FRONTEND=noninteractive
apt_update_cmd="apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 -o Acquire::CompressionTypes::Order::=gz -o Acquire::AllowInsecureRepositories=true -o APT::Get::AllowUnauthenticated=true update"
$apt_update_cmd
install_pkgs=""

append_pkg() {
  pkg="$1"
  case " $install_pkgs " in
    *" $pkg "*) ;;
    *) install_pkgs="${install_pkgs} ${pkg}" ;;
  esac
}

probe_first_available() {
  logical_name="$1"
  shift
  resolved=""
  for candidate in "$@"; do
    if apt-cache show "$candidate" 2>/dev/null | grep -q '^Package:'; then
      resolved="$candidate"
      break
    fi
  done

  if [ -n "$resolved" ]; then
    append_pkg "$resolved"
    echo "resolved: $logical_name -> $resolved"
    return 0
  fi

  echo "missing: $logical_name" >&2
  return 1
}

probe_first_available build-essential build-essential || true
probe_first_available pkg-config pkg-config pkgconf || true
probe_first_available autoconf autoconf || true
probe_first_available automake automake || true
probe_first_available libtool libtool libtool-bin || true
probe_first_available cmake cmake cmake3 || true
probe_first_available ninja-build ninja-build ninja || true
probe_first_available meson meson || true
probe_first_available git git git-core || true
probe_first_available ca-certificates ca-certificates || true
probe_first_available python3 python3 python2 python python2.7 python2.6 python2.5 || true
probe_first_available python3-dev python3-dev python-dev python2-dev python2.7-dev python2.6-dev python2.5-dev || true
probe_first_available libssl-dev libssl-dev libssl1.0-dev libssl0.9.8-dev || true

if [ -z "$install_pkgs" ]; then
  echo "no common build dependencies could be resolved" >&2
  exit 1
fi

attempt=1
max_attempts=5
while :; do
  # shellcheck disable=SC2086
  if apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 -o APT::Get::AllowUnauthenticated=true install -y --allow-unauthenticated --force-yes --fix-missing $install_pkgs; then
    break
  fi
  if [ "$attempt" -ge "$max_attempts" ]; then
    echo "apt-get install failed after ${max_attempts} attempts" >&2
    exit 1
  fi
  attempt=$((attempt + 1))
  echo "retrying apt-get install (${attempt}/${max_attempts})" >&2
  $apt_update_cmd
  sleep 5
done
EOF

CMD ["/bin/bash"]
