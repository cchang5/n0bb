import sys
sys.path.append('/Users/cchang5/Physics/c51/scripts')
import numpy as np
import lsqfit
import gvar as gv
import tqdm
import subprocess
import psycopg2 as psql
import psycopg2.extras as psqle
import password_file as pwd

def login(nerscusr,sqlusr,pwd):
    subprocess.call('ssh -fCN %s@edison.nersc.gov -o TCPKeepAlive=yes -L 5555:scidb1.nersc.gov:5432' %(nerscusr), shell=True)
    hostname='localhost'
    databasename='c51_project2'
    portnumber='5555'
    try:
        conn = psql.connect(host=hostname, port=portnumber, database=databasename, user=sqlusr, password=pwd)
        cur = conn.cursor()
        dict_cur = conn.cursor(cursor_factory=psqle.RealDictCursor)
        user = sqlusr
        print "connected"
        return cur, conn
    except psql.Error as e:
        print "unable to connect to DB"
        print e
        print e.pgcode
        print e.pgerror
        print traceback.format_exc()
        print "exiting"
        raise SystemExit

if __name__=="__main__":
    psqlpwd = pwd.passwd()
    cur, conn = login('cchang5','cchang5',psqlpwd)
