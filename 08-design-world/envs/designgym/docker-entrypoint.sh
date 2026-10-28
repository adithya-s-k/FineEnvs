#!/bin/sh
# The task pack: the one at DESIGN_CANVAS_PACK, or downloaded there from the Hub the first time.
PACK="${DESIGN_CANVAS_PACK:-/data/pack}"
if [ ! -f "$PACK/manifest.json" ]; then
  designgym pull --pack "$PACK" ${DESIGNGYM_PACK_SPLITS:+--splits "$DESIGNGYM_PACK_SPLITS"} || exit 1
fi
export DESIGN_CANVAS_PACK="$PACK"
exec "$@"
