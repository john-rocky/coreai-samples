#!/bin/zsh
# Zoo mode: clone apple/coreai-models locally, apply the zoo's engine patches,
# and regenerate CoreAIChatMac against the patched checkout.
#
# Patch source of truth = the zoo repo (Apple's repo accepts no PRs, so engine
# extensions ship as patch files there). Pinned commits keep this reproducible:
# bump APPLE_PIN/ZOO_PIN together after testing a newer combination.
set -e
cd "$(dirname "$0")/.."

APPLE_PIN=b1cb71b   # apple/coreai-models revision the patches are verified against
ZOO_PIN=f941494     # john-rocky/coreai-model-zoo revision the patches come from
ZOO_RAW="https://raw.githubusercontent.com/john-rocky/coreai-model-zoo/${ZOO_PIN}/apps"
CLONE=zoo-coreai-models
# Order matters: per-token-inputs / static-inputs apply on top of extra-states.
PATCHES=(
  coreai-shared-product
  coreai-pipelined-extra-states
  coreai-pipelined-per-token-inputs
  coreai-pipelined-static-inputs
)

if [ ! -d "$CLONE" ]; then
  git clone https://github.com/apple/coreai-models "$CLONE"
fi
git -C "$CLONE" checkout -q "$APPLE_PIN"

for p in $PATCHES; do
  echo "applying ${p}.patch"
  curl -fsSL "$ZOO_RAW/${p}.patch" -o "/tmp/${p}.patch"
  if git -C "$CLONE" apply --check "/tmp/${p}.patch" 2>/dev/null; then
    git -C "$CLONE" apply "/tmp/${p}.patch"
  else
    echo "  -> already applied or conflicts; skipping (run 'git -C $CLONE status' to inspect)"
  fi
done

cd CoreAIChatMac
xcodegen generate --spec project-zoo.yml
echo ""
echo "Zoo mode ready: open CoreAIChatMac/CoreAIChatMac.xcodeproj"
echo "(to go back to the official build: xcodegen generate --spec project.yml)"
