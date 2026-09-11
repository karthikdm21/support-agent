import duckdb
con = duckdb.connect()
top_brands = con.execute("""
    SELECT author_id, COUNT(*) as cnt
    FROM read_csv_auto('data/twcs.csv')
    WHERE inbound = FALSE
    GROUP BY author_id
    ORDER BY cnt DESC
    LIMIT 15
""").df()
print(top_brands)