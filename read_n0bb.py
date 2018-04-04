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
    # matrix elements
    if False:
        files = ['./l1648/l1648mpi310_fix_rat.dat','./l2448/l2448mpi220_fix_rat.dat','./l2464mpi220/l2464mpi220_fix_rat.dat','./l2464mpi310/l2464mpi310_fix_rat.dat','./l3248/l3248mpi135_fix_rat.dat','./l3264/l3264mpi220_fix_rat.dat','./l3296/l3296mpi310_fix_rat.dat','./l4064/l4064mpi220_fix_rat.dat','./l4864/l4864mpi135_fix_rat.dat','./l4896/l4896mpi220_fix_rat.dat']
        ens_list = ['l1648f211b580m013m065m838','l2448f211b580m0064m0640m828','l2464f211b600m00507m0507m628','l2464f211b600m0102m0509m635','l3248f211b580m00235m0647m831','l3264f211b600m00507m0507m628','l3296f211b630m0074m037m440','l4064f211b600m00507m0507m628','l4864f211b600m00184m0507m628','l4896f211b630m00363m0363m430']
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
    # epsilon pi
    if False:
        files = ['./data_v2/l1648_eps.dat','./data_v2/l2448_eps.dat','./data_v2/l2464_220_eps.dat','./data_v2/l2464_310_eps.dat','./data_v2/l3264_eps.dat','./data_v2/l3296_eps.dat','./data_v2/l4064_eps.dat','./data_v2/l4864_eps.dat','./data_v2/l4896_eps.dat']
        ens_list = ['l1648f211b580m013m065m838','l2448f211b580m0064m0640m828','l2464f211b600m00507m0507m628','l2464f211b600m0102m0509m635','l3264f211b600m00507m0507m628','l3296f211b630m0074m037m440','l4064f211b600m00507m0507m628','l4864f211b600m00184m0507m628','l4896f211b630m00363m0363m430']
        for idx, f in enumerate(files):
            ens = ens_list[idx]
            print ens, f
            with open(f,'r') as d:
                op = 'epi'
                mean = []
                for nbs,l in tqdm.tqdm(enumerate(d)):
                    temp = l.split(',')
                    mean_n = []
                    for fit_n,t in enumerate(temp):
                        data = float(t.replace('\n',''))
                        mean_n.append(data)
                        sqlcmd = "SELECT callat_fcn.upsert($$INSERT INTO callat_proj.n0bb_v2 (hisq_ensembles, operator, nbs, fit_n, result) VALUES ('%s','%s',%s,%s,%s)$$);" %(ens,op,nbs+1,fit_n,data)
                        cur.execute(sqlcmd)
                        conn.commit()
                    mean.append(mean_n)
                boot0 = np.mean(mean,axis=0)
                for fit_n,data in enumerate(boot0):
                    sqlcmd = "SELECT callat_fcn.upsert($$INSERT INTO callat_proj.n0bb_v2 (hisq_ensembles, operator, nbs, fit_n, result) VALUES ('%s','%s',%s,%s,%s)$$);" %(ens,op,0,fit_n,data)
                    cur.execute(sqlcmd)
                    conn.commit()
    # fpi
    if True:
        files = ['./data_v1/l1648_fpi_old.dat','./data_v1/l2448_fpi_old.dat','./data_v1/l2464_220_fpi_old.dat','./data_v1/l2464_310_fpi_old.dat','./data_v1/l3264_fpi_old.dat','./data_v1/l3296_fpi_old.dat','./data_v1/l4064_fpi_old.dat','./data_v1/l4864_fpi_old.dat','./data_v1/l4896_fpi_old.dat']
        ens_list = ['l1648f211b580m013m065m838','l2448f211b580m0064m0640m828','l2464f211b600m00507m0507m628','l2464f211b600m0102m0509m635','l3264f211b600m00507m0507m628','l3296f211b630m0074m037m440','l4064f211b600m00507m0507m628','l4864f211b600m00184m0507m628','l4896f211b630m00363m0363m430']
        for idx, f in enumerate(files):
            ens = ens_list[idx]
            print ens, f
            with open(f,'r') as d:
                op = 'fpi'
                mean = []
                for nbs,l in tqdm.tqdm(enumerate(d)):
                    temp = l.split(',')
                    mean_n = []
                    for fit_n,t in enumerate(temp):
                        data = float(t.replace('\n',''))
                        mean_n.append(data)
                        sqlcmd = "SELECT callat_fcn.upsert($$INSERT INTO callat_proj.n0bb_v1 (hisq_ensembles, operator, nbs, fit_n, result) VALUES ('%s','%s',%s,%s,%s)$$);" %(ens,op,nbs+1,fit_n,data)
                        cur.execute(sqlcmd)
                        conn.commit()
                    mean.append(mean_n)
                boot0 = np.mean(mean,axis=0)
                for fit_n,data in enumerate(boot0):
                    sqlcmd = "SELECT callat_fcn.upsert($$INSERT INTO callat_proj.n0bb_v1 (hisq_ensembles, operator, nbs, fit_n, result) VALUES ('%s','%s',%s,%s,%s)$$);" %(ens,op,0,fit_n,data)
                    cur.execute(sqlcmd)
                    conn.commit()

