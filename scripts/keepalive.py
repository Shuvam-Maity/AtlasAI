from datetime import datetime

from src.graph_db import run_query, close

rows = run_query("MATCH (e:Entity) RETURN count(e) AS entities")
print(f"{datetime.now():%Y-%m-%d %H:%M}  instance alive, {rows[0]['entities']} entities")
close()