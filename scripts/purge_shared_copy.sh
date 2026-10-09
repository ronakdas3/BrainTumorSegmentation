#!/bin/bash
# Purge this conversation from the shared public directory
SHARED_DIR="/home/ub/.gemini/antigravity"
CONV_ID="8be8c6f5-3d41-4736-8b83-1f52f29957be"

echo "Purging session $CONV_ID from shared store..."

# 1. Remove from shared conversation summaries
if [ -f "$SHARED_DIR/conversation_summaries.db" ]; then
    sqlite3 "$SHARED_DIR/conversation_summaries.db" "DELETE FROM conversation_summaries WHERE conversation_id='$CONV_ID';" 2>/dev/null
    echo "  - Removed from shared conversation_summaries.db"
fi

# 2. Remove shared conversation database
rm -f "$SHARED_DIR/conversations/${CONV_ID}"*
echo "  - Removed shared database files"

# 3. Remove shared brain directory
rm -rf "$SHARED_DIR/brain/${CONV_ID}"
echo "  - Removed shared brain directory"

echo "✅ Done! Shared directory is now completely clean of this session."
