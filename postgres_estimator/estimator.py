import time
import numpy as np
class CardEst(object):
    """Base class for a cardinality estimator."""

    def __init__(self):
        self.query_starts = []
        self.query_dur_ms = []
        self.errs = []
        self.est_cards = []
        self.true_cards = []

        self.name = 'CardEst'

    def Query(self, columns, operators, vals):
        """Estimates cardinality with the specified conditions.

        Args:
            columns: list of Column objects to filter on.
            operators: list of string representing what operation to perform on
              respective columns; e.g., ['<', '>='].
            vals: list of raw values to filter columns on; e.g., [50, 100000].
              These are not bin IDs.
        Returns:
            Predicted cardinality.
        """
        raise NotImplementedError

    def OnStart(self):
        self.query_starts.append(time.time())

    def OnEnd(self):
        self.query_dur_ms.append((time.time() - self.query_starts[-1]) * 1e3)

    def AddError(self, err):
        self.errs.append(err)

    def AddError(self, err, est_card, true_card):
        self.errs.append(err)
        self.est_cards.append(est_card)
        self.true_cards.append(true_card)

    def __str__(self):
        return self.name

    def get_stats(self):
        return [
            self.query_starts, self.query_dur_ms, self.errs, self.est_cards,
            self.true_cards
        ]

    def merge_stats(self, state):
        self.query_starts.extend(state[0])
        self.query_dur_ms.extend(state[1])
        self.errs.extend(state[2])
        self.est_cards.extend(state[3])
        self.true_cards.extend(state[4])

    def report(self):
        est = self
        if est.name == 'CardEst':
            est.name = str(est)
        print(est.name, "max", np.max(est.errs), "99th",
              np.quantile(est.errs, 0.99), "95th", np.quantile(est.errs, 0.95),
              "median", np.quantile(est.errs, 0.5), "time_ms",
              np.mean(est.query_dur_ms))

def QueryToPredicate(columns, operators, vals, wrap_as_string_cols=None):
    """Converts from (c,o,v) to sql string (for Postgres)."""
    v_s = [
        str(v).replace('T', ' ') if type(v) is np.datetime64 else v
        for v in vals
    ]
    v_s = ["\'" + v + "\'" if type(v) is str else str(v) for v in v_s]

    if wrap_as_string_cols is not None:
        for i in range(len(columns)):
            if columns[i].name in wrap_as_string_cols:
                v_s[i] = "'" + str(v_s[i]) + "'"

    preds = [
        c.pg_name + ' ' + o + ' ' + v
        for c, o, v in zip(columns, operators, v_s)
    ]
    s = ' and '.join(preds)
    return ' where ' + s

class Postgres(CardEst):
    def __init__(self, database, table_names, port=None, user='imdb', password='password'):
        """Postgres estimator (i.e., EXPLAIN).  Must have the PG server live.
        E.g.,
            def MakeEstimators():
                return [Postgres('dmv', 'vehicle_reg', None), ...]
        Args:
          database: string, the database name.
          table_names: List[string], list of table names
          port: int, the port.
        """
        import psycopg2

        super(Postgres, self).__init__()

        self.conn = psycopg2.connect(user=user, password=password,
                                     database=database, port=port, host='localhost')
        self.conn.autocommit = True
        self.cursor = self.conn.cursor()
        self.name = 'Postgres'

        for relation in table_names:
            self.cursor.execute('analyze ' + relation + ';')
            self.conn.commit()
        self.table_names = table_names

        self.database = database

    def __str__(self):
        return 'postgres'

    def QuerySql(self, sql_query):
        """
        Args:
            sql_query (string): sql query to run
        """
        sql_query = sql_query.replace('COUNT(*)', '*')
        sql_query = sql_query.replace('SELECT(*)', 'SELECT *')
        query_s = 'explain(format json) ' + sql_query
        # print(query_s)
        self.OnStart()
        self.cursor.execute(query_s)
        res = self.cursor.fetchall()
        # print(res)
        result = res[0][0][0]['Plan']['Plan Rows']
        self.OnEnd()
        return result

    def QueryByExecSql(self, sql_query):
        query_s = sql_query
        self.string = query_s

        self.cursor.execute(query_s)
        result = self.cursor.fetchone()[0]

        return result

    def Query(self, columns, operators, vals):
        assert len(columns) == len(operators) == len(vals)
        pred = QueryToPredicate(columns, operators, vals)
        # Use json so it's easier to parse.
        query_s = 'select * from ' + self.relation + pred
        return self.QuerySql(query_s)

    def QueryByExec(self, columns, operators, vals):
        # Runs actual query on postgres and returns true cardinality.
        assert len(columns) == len(operators) == len(vals)

        pred = QueryToPredicate(columns, operators, vals)
        query_s = 'select count(*) from ' + self.relation + pred
        return self.QueryByExecSql(query_s)

    def Close(self):
        self.cursor.close()
        self.conn.close()