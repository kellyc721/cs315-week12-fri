# Short script to create a file with submission posts for testing Topic Modeling

import duckdb
DB_PATH = "kdrama_reddit.duckdb"

query = """
CREATE temp table filtered_submissions AS (
    SELECT 
        timestamp AS timestamp,
        subreddit,
        'submission' AS type,
        post_id AS id,
        title AS submission_title,
        selftext AS body
    FROM submissions
    WHERE selftext IS NOT NULL 
      AND len(string_split(trim(selftext), ' ')) BETWEEN 20 AND 400
)
"""

with duckdb.connect(DB_PATH) as con:
    con.execute(query)
    con.execute("COPY filtered_submissions TO 'filtered_submissions.parquet' (FORMAT PARQUET)")

    counts = dict(con.execute("SELECT type, COUNT(*) FROM filtered_submissions GROUP BY type").fetchall())
    print(f"Submissions selected: {counts.get('submission', 0):,}")