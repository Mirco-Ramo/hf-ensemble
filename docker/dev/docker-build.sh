#!/usr/bin/env bash
set -e

command=$(basename $0)


function print_help {
    echo "Usage: $command VERSION [-c COMMIT|BRANCH] [--cache]"
    echo "Build the Docker image tags it with the specified version".
    echo
    echo "Mandatory arguments:"
    echo "  VERSION             the release version number"
    echo "  -c, --checkout      checkout the specified git commit or branch, by default use the version's tag"
    echo
    echo "Optional arguments:"
    echo "      --cache         build Dockerfile using Docker cache"
}

DOCKER_OPS="--no-cache"
while [[ $# -gt 0 ]]; do
    arg_key="$1"

    case ${arg_key} in
        -c|--checkout)
        CHECKOUT="$2"
        shift # past argument
        shift # past value
        ;;
        --cache)
        DOCKER_OPS=""
        shift # past argument
        ;;
        -h|--help)
        print_help
        exit 0
        ;;
        *)    # unknown option
        if [ -z "$VERSION" ]; then
          VERSION="$1"
          shift
        else
          echo "$command: invalid option '$arg_key'"
          echo "Try '$command --help' for more information"
          exit 1
        fi
        ;;
    esac
done

if [ -z "$VERSION" ]; then
    echo "$command: missing operand"
    echo "Try '$command --help' for more information."
    exit 1
fi

if [ -z "$CHECKOUT" ]; then
  CHECKOUT="v$VERSION"
fi


docker build $DOCKER_OPS \
    --build-arg GIT_TOKEN=${GIT_TOKEN} . \
    -t "hf/dev_ensemble:${VERSION}" -t hf/dev_ensemble:latest
