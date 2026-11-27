"""Mini code lab: SQL (in-memory SQLite, read-only) and Python (subprocess + timeout)."""
import json
import sqlite3
import subprocess
import sys
import time

SQL_SCHEMA = """
CREATE TABLE employees(id INTEGER, name TEXT, dept TEXT, salary INTEGER);
INSERT INTO employees VALUES (1,'Asha','Data',70000),(2,'Ravi','Data',55000),
 (3,'Meera','HR',48000),(4,'Kabir','HR',52000),(5,'Isha','Dev',90000),(6,'Neel','Dev',45000);
CREATE TABLE customers(id INTEGER, name TEXT, city TEXT);
INSERT INTO customers VALUES (1,'Anil','Kanpur'),(2,'Sara','Delhi'),(3,'Tina','Pune');
CREATE TABLE orders(id INTEGER, customer_id INTEGER, amount INTEGER);
INSERT INTO orders VALUES (1,1,500),(2,1,300),(3,2,700),(4,2,100),(5,1,200);
"""
SQL_TABLES_DOC = ("employees(id, name, dept, salary)\n"
                  "customers(id, name, city)\n"
                  "orders(id, customer_id, amount)")

SQL_CHALLENGES = [
    dict(id="sql1", title="High earners", kind="SQL",
         prompt="Return the names of employees earning more than 50000.",
         ref="SELECT name FROM employees WHERE salary > 50000"),
    dict(id="sql2", title="Average salary per department", kind="SQL",
         prompt="Return dept and avg_salary for each department, ordered by dept.",
         ref="SELECT dept, AVG(salary) AS avg_salary FROM employees GROUP BY dept ORDER BY dept"),
    dict(id="sql3", title="Customer totals", kind="SQL",
         prompt="Return customer name and total order amount (column: total), "
                "highest first. Only customers who placed orders.",
         ref="SELECT c.name, SUM(o.amount) AS total FROM customers c "
             "JOIN orders o ON o.customer_id = c.id GROUP BY c.name ORDER BY total DESC"),
    dict(id="sql4", title="Customer with the most orders", kind="SQL",
         prompt="Return the customer_id that placed the largest number of orders.",
         ref="SELECT customer_id FROM orders GROUP BY customer_id "
             "ORDER BY COUNT(*) DESC LIMIT 1"),
]

PY_CHALLENGES = [
    dict(id="py1", title="Two Sum", kind="Python", fn="two_sum",
         prompt="Return indices of the two numbers that add up to target.",
         starter="def two_sum(nums, target):\n    pass\n",
         tests=[([[2, 7, 11, 15], 9], [0, 1]), ([[3, 2, 4], 6], [1, 2]), ([[3, 3], 6], [0, 1])]),
    dict(id="py2", title="Valid Palindrome", kind="Python", fn="is_palindrome",
         prompt="True if the string is a palindrome ignoring case and non-alphanumerics.",
         starter="def is_palindrome(s):\n    pass\n",
         tests=[(["A man, a plan, a canal: Panama"], True), (["race a car"], False), ([""], True)]),
    dict(id="py3", title="Maximum Subarray", kind="Python", fn="max_subarray",
         prompt="Return the largest sum of any contiguous subarray.",
         starter="def max_subarray(nums):\n    pass\n",
         tests=[([[-2, 1, -3, 4, -1, 2, 1, -5, 4]], 6), ([[1]], 1), ([[-3, -1, -2]], -1)]),
]


def _norm(rows):
    return [tuple(round(x, 2) if isinstance(x, float) else x for x in r) for r in rows]


def _run_sql_raw(query):
    con = sqlite3.connect(":memory:")
    con.executescript(SQL_SCHEMA)
    con.execute("PRAGMA query_only = ON")
    cur = con.execute(query)
    cols = [d[0] for d in cur.description or []]
    return cols, _norm(cur.fetchall())


def run_sql(query, ch):
    q = query.strip().rstrip(";")
    if not q.lower().startswith(("select", "with")):
        return dict(ok=False, error="Only SELECT / WITH queries are allowed.", passed=0, total=1)
    try:
        cols, rows = _run_sql_raw(q)
        _, ref_rows = _run_sql_raw(ch["ref"])
    except Exception as e:
        return dict(ok=False, error=str(e), passed=0, total=1)
    ordered = "order by" in ch["ref"].lower()
    same = (rows == ref_rows) if ordered else (sorted(rows, key=str) == sorted(ref_rows, key=str))
    return dict(ok=True, error=None, columns=cols, rows=rows[:50],
                passed=int(same), total=1)


HARNESS = '''
import json, sys
_tests = json.loads(sys.stdin.read())
_res = []
for _a, _e in _tests:
    try:
        _r = FN(*_a)
        _ok = (_r == _e) or (isinstance(_r, (list, tuple)) and isinstance(_e, list)
                             and sorted(_r) == sorted(_e))
    except Exception:
        _ok = False
    _res.append(bool(_ok))
sys.stdout.write("\\n@@RESULT@@" + json.dumps(_res))
'''


def run_python(code, ch, timeout=5):
    """Runs user code in a separate process with a timeout.
    NOTE: not a hardened sandbox. Do not expose publicly without containerization."""
    src = code + "\n" + HARNESS.replace("FN", ch["fn"])
    t0 = time.perf_counter()
    try:
        p = subprocess.run([sys.executable, "-I", "-c", src],
                           input=json.dumps(ch["tests"]), capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return dict(ok=False, error=f"Time limit exceeded ({timeout}s)", passed=0,
                    total=len(ch["tests"]), ms=timeout * 1000)
    ms = (time.perf_counter() - t0) * 1000
    total = len(ch["tests"])
    if "@@RESULT@@" not in p.stdout:
        return dict(ok=False, error=(p.stderr or "No result")[-600:], passed=0,
                    total=total, ms=ms)
    res = json.loads(p.stdout.split("@@RESULT@@")[1])
    return dict(ok=True, error=None, passed=sum(res), total=total, detail=res, ms=ms)
