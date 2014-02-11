# Manager for database
# Handles insertion of np.ndarrays
# Returns results as namedtuples

import io
import numpy as np
import sqlite3
from collections import namedtuple

def namedtuple_row_factory(cursor, row):
    fields = [col[0] for col in cursor.description]
    Row = namedtuple("row", fields)
    return Row(*row)

def array_to_buffer(arr):
    '''Serialize np.ndarray'''
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return buffer(out.read())

def buffer_to_array(text):
    '''Marshall serialized np.ndarray'''
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


class DB_Manager(object):
    '''Handles Numpy Adapters, serialization etc.'''
    sqlite3.register_adapter(np.ndarray, array_to_buffer)
    sqlite3.register_converter("array", buffer_to_array)

    DB_Info = namedtuple("DB_Info", ('src',))
    _default_db_info = DB_Info('data.db',)

    def __init__(self, **kwargs):
        _db_info = kwargs.get('db_info', DB_Manager._default_db_info)
        self.connections = {}
        for name, f in _db_info.iteritems():
            conn = sqlite3.connect(f, check_same_thread=False)
            conn.row_factory = namedtuple_row_factory
            self.connections[name] = conn

    def prepare_cursor(self, db_name, q, opts):
        '''Returns executed cursor'''
        conn = self.connections[db_name]
        cur = conn.cursor()
        if sqlite3.complete_statement(q):
            q = q.strip()
            cur.execute(q, opts)
        else:
            raise ValueError('""%s"" is not a valid SQL Statement' % q)
        return cur

    def query(self, db_name, q, opts, commit=False):
        cur = self.prepare_cursor(db_name, q, opts)
        results = cur.fetchall()
        if commit:
            self.commit(db_name)
        return results

    def close_all(self):
        for k in self.connections:
            self.connections[k].close()

    def commit(self, db_name):
        return self.connections[db_name].commit()
