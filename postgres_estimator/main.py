from mako import template
import pandas as pd
import textwrap
import csv
from estimator import Postgres
import time
import argparse

def row_to_query(row):
    """convert rows in csv file into real sql query
        input: row: row in pandas df, df's names = ['Tables', 'Joins', 'Filters', 'Cardinality']
    """
    template_full = template.Template(
    textwrap.dedent("""
    SELECT COUNT(*)
    FROM ${table_names}
    WHERE ${join_clauses}
    AND ${filter_clauses};
""").strip())
    template_one_clauses = template.Template(
        textwrap.dedent("""
        SELECT COUNT(*)
        FROM ${table_names}
        WHERE ${clauses};
    """).strip())
    template_no_clauses = template.Template(
        textwrap.dedent("""
        SELECT COUNT(*)
        FROM ${table_names};
    """).strip())
    str_cols = ['country_code', 'phonetic_code', 'season_nr', 'name', 'imdb_index', 'name_pcode_cf', 'name_pcode_nf', 'surname_pcode',
            'md5sum', 'title', 'imdb_index', 'note', 'kind', 'link', 'gender', 'role', 'series_years', 'info']
    
    joins = row['Joins'].split(',') if len(row['Joins'])>0 else []
    join_clauses = '\n AND '.join(joins)
    filters = row['Filters'].split(',') if len(row['Filters'])>0 else []
    filter_clauses = '\n AND '.join([
                    '{} {} {}'.format(filters[i], filters[i+1], filters[i+2])
                        if filters[i].split('.')[-1] not in str_cols else
                    '{} {} \'{}\''.format(filters[i], filters[i+1], filters[i+2])
                for i in range(0, len(filters), 3)
            ])
    sql = None
    table_names = row['Tables']
    if len(join_clauses) > 0 and len(filter_clauses)>0:
        sql = template_full.render(table_names=table_names,
                                                join_clauses=join_clauses,
                                                filter_clauses=filter_clauses)
    elif len(join_clauses) > 0 and len(filter_clauses)==0:
        sql = template_one_clauses.render(table_names=table_names,
                                                clauses=join_clauses)
    elif len(join_clauses) == 0 and len(filter_clauses) > 0:
        sql = template_one_clauses.render(table_names=table_names,
                                                clauses=filter_clauses)
    else:
        sql = template_no_clauses.render(table_names=table_names)
    return sql

def get_true_card(estimator, sql):
    """
    input: sql: str, sql qeury
    """
    return estimator.QueryByExecSql(sql)

def get_estimated_card(estimator, sql):
    """
    input: sql: str, sql qeury
    """
    return estimator.QuerySql(sql)

def get_est_card_with_csv(f_csv, path_output, sep='#', database='imdb', user='imdb', password='password', port=5432,
                          table_names=['title', 'cast_info', 'movie_info', 'movie_companies', 'movie_keyword',
             'movie_info_idx']):
    df = pd.read_csv(f_csv, sep=sep, names=['Tables', 'Joins', 'Filters', 'Cardinality'])
    df['Joins'] = df['Joins'].fillna('')
    df['Filters'] = df['Filters'].fillna('')
    emt = Postgres('imdb', table_names, port=port)
    estimated_cards = []
    start_time = time.time()
    for idx, row in df.iterrows():
        sql = row_to_query(row)
        estimated_cards.append(get_estimated_card(emt, sql))
    execution_time = time.time()-start_time
    print(f"Postgres estiamted {len(estimated_cards)} rows within {execution_time} seconds")
    df['postgres_estimated_card'] = estimated_cards
    df.to_csv(path_output, sep='#', index=False, quoting=csv.QUOTE_NONE)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--f_csv', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--path_output', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--sep', type=str, default='#', help='CSV separator')
    parser.add_argument('--database', type=str, default='imdb', help='Database name')
    parser.add_argument('--user', type=str, default='imdb', help='Database user')
    parser.add_argument('--password', type=str, default='password', help='Database password')
    parser.add_argument('--port', type=int, default=5432, help='Database port')
    args = parser.parse_args()
    get_est_card_with_csv(
        f_csv=args.f_csv,
        path_output=args.path_output,
        sep=args.sep,
        database=args.database,
        user=args.user,
        password=args.password,
        port=args.port
    )

if __name__ == '__main__':
    main()