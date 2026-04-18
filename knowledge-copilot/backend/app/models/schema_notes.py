"""
MongoDB Collections — Knowledge Copilot
========================================

Collection: users
-----------------
{
  "_id":          ObjectId,          # MongoDB primary key
  "email":        str,               # unique index
  "name":         str,
  "password_hash": str | None,       # None for OAuth-only users
  "auth_provider": "email" | "google" | "clerk",
  "clerk_user_id": str | None,       # set when provider = clerk/google
  "avatar_url":   str | None,
  "is_active":    bool,              # soft disable without deletion
  "is_verified":  bool,              # email verification flag
  "created_at":   datetime,
  "updated_at":   datetime,
  "last_login":   datetime | None,
}

Indexes:
  - email: unique
  - clerk_user_id: sparse unique (only for clerk users)

Collection: chat_sessions
--------------------------
{
  "_id":        ObjectId,
  "user_id":    ObjectId,            # ref → users._id
  "title":      str,                 # auto-generated from first message
  "created_at": datetime,
  "updated_at": datetime,
  "is_active":  bool,
}

Indexes:
  - user_id + created_at (compound, for listing sessions by user)

Collection: chat_messages
--------------------------
{
  "_id":        ObjectId,
  "session_id": ObjectId,            # ref → chat_sessions._id
  "user_id":    ObjectId,            # denormalised for fast queries
  "role":       "user" | "assistant",
  "content":    str,
  "sources": [                       # populated for assistant messages
    {
      "file_name": str,
      "page":      int | None,
      "score":     float,
    }
  ],
  "created_at": datetime,
}

Indexes:
  - session_id + created_at (compound)
  - user_id (for user-level history queries)

Design notes:
  - Messages are a separate collection (not embedded) so sessions with
    hundreds of messages don't hit MongoDB's 16 MB document size limit.
  - user_id is denormalised onto messages so you can query
    "all messages by user X" without a join through sessions.
  - The existing in-memory session store from Phase 7 is replaced by
    these collections once auth is wired in.
"""