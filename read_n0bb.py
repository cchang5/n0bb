import sys
sys.path.append('/Users/cchang5/Physics/c51/scripts')
import numpy as np
import password_file as pwd
import tqdm
import subprocess
import psycopg2 as psql
import psycopg2.extras as psqle

def read_op(d):
    data = d.readline().replace("\"",'').split("},{")
    data[0] = data[0].replace("{",'')
    data[-1] = data[-1].replace("}",'').replace("\n",'')
    data = np.array(data)
    temp = []
    for idx,i in enumerate(data):
        t = [float(j) for j in i.split(',')]
        temp.append(t)
    data = np.array(temp)
    return data

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

if __name__=='__main__':
    psqlpwd = pwd.passwd()
    cur, conn = login('cchang5','cchang5',psqlpwd)    
    files = ['l1648mpi200_final.dat','l2464mpi220_final.dat','l3248mpi135_final.dat','l3296mpi220_final.dat','l4864mpi220_final.dat','l2448mpi220_final.dat','l2464mpi310_final.dat','l3264mpi220_final.dat','l4064mpi220_final.dat','l4896mpi220_final.dat','l3248mpi135_final.dat']
    ens_list = ['l1648f211b580m013m065m838','l2464f211b600m00507m0507m628','l3248f211b580m00235m0647m831','l3296f211b630m0074m037m440','l4864f211b600m00184m0507m628','l2448f211b580m0064m0640m828','l2464f211b600m0102m0509m635','l3264f211b600m00507m0507m628','l4064f211b600m00507m0507m628','l4896f211b630m00363m0363m430','l3248f211b580m00235m0647m831']
    #files = ['l1648mpi200_final.dat']
    #ens_list = ['l1648f211b580m013m065m838']
    for idx, f in enumerate(files):
        ens = ens_list[idx]
        print ens, f
        with open(f,'r') as d:
            for o in tqdm.tqdm(range(5),desc='op',leave=True):
                if o == 0:
                    op = 'LR'
                elif o == 1:
                    op = 'S'
                elif o == 2:
                    op = 'V'
                elif o == 3:
                    op = 'LR_colormix'
                elif o == 4:
                    op = 'S_colormix'
                data = read_op(d)
                for fit_n in range(1,len(data[0])): # write boot0
                    sqlcmd = "SELECT callat_fcn.upsert($$INSERT INTO callat_proj.n0bb_v1 (hisq_ensembles, operator, nbs, fit_n, result) VALUES ('%s','%s',%s,%s,%s)$$);" %(ens,op,0,fit_n,data[0,0])
                    cur.execute(sqlcmd)
                    conn.commit()
                for nbs in tqdm.tqdm(range(len(data)),desc='nbs',leave=True,nested=True):
                    for fit_n in range(1,len(data[0])):
                        sqlcmd = "SELECT callat_fcn.upsert($$INSERT INTO callat_proj.n0bb_v1 (hisq_ensembles, operator, nbs, fit_n, result) VALUES ('%s','%s',%s,%s,%s)$$);" %(ens,op,nbs+1,fit_n,data[nbs,fit_n])
                        cur.execute(sqlcmd)
                        conn.commit()
