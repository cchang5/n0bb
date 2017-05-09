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
import matplotlib.pyplot as plt

def params():
    # set up data set
    p = dict()
    #p['ens'] = ['l1648f211b580m013m065m838','l2448f211b580m0064m0640m828','l2464f211b600m00507m0507m628','l2464f211b600m0102m0509m635','l3248f211b580m00235m0647m831','l3264f211b600m00507m0507m628','l3296f211b630m0074m037m440','l4064f211b600m00507m0507m628','l4864f211b600m00184m0507m628','l4896f211b630m00363m0363m430']
    p['ens'] = ['l1648f211b580m013m065m838','l2448f211b580m0064m0640m828','l2464f211b600m00507m0507m628','l2464f211b600m0102m0509m635','l3264f211b600m00507m0507m628','l3296f211b630m0074m037m440','l4064f211b600m00507m0507m628','l4864f211b600m00184m0507m628','l4896f211b630m00363m0363m430']
    p['fit_n'] = [1]
    p['op'] = ['LR','S','V','LR_colormix','S_colormix']
    #p['op'] = ['S']
    #p['op'] = ['V']
    # list pion data indices
    p['mes_idx'] = dict()
    p['mes_idx']['l1648f211b580m013m065m838']    = {'mpi': 47, 'mres': 26, 'mbs': 1960}
    p['mes_idx']['l2448f211b580m0064m0640m828']  = {'mpi': 48, 'mres': 27, 'mbs': 1000}
    p['mes_idx']['l3248f211b580m00235m0647m831'] = {'mpi': 40, 'mres': 20, 'mbs': 1000}
    p['mes_idx']['l2464f211b600m0102m0509m635']  = {'mpi': 27, 'mres': 29, 'mbs': 1053}
    p['mes_idx']['l2464f211b600m00507m0507m628'] = {'mpi': 46, 'mres': 25, 'mbs': 1000}
    p['mes_idx']['l3264f211b600m00507m0507m628'] = {'mpi': 26, 'mres': 13, 'mbs': 1000}
    p['mes_idx']['l4064f211b600m00507m0507m628'] = {'mpi': 45, 'mres': 24, 'mbs': 1000}
    p['mes_idx']['l4864f211b600m00184m0507m628'] = {'mpi': 24, 'mres': 14, 'mbs': 1000}
    p['mes_idx']['l3296f211b630m0074m037m440']   = {'mpi': 49, 'mres': 28, 'mbs': 784}
    p['mes_idx']['l4896f211b630m00363m0363m430'] = {'mpi': 28, 'mres': 11, 'mbs': 1001}
    # list priors
    p['prior'] = dict()
    p['prior']['g.LR'] = [0.0, 10.0]
    p['prior']['c.LR'] = [0.0, 10.0]
    p['prior']['a.LR'] = [0.0, 10.0]
    p['prior']['m.LR'] = [0.0, 10.0]
    p['prior']['g.S'] = [0.0, 10.0]
    p['prior']['c.S'] = [0.0, 10.0]
    p['prior']['a.S'] = [0.0, 10.0]
    p['prior']['m.S'] = [0.0, 10.0]
    p['prior']['g.V'] = [0.0, 10.0]
    p['prior']['c.V'] = [0.0, 10.0]
    p['prior']['a.V'] = [0.0, 10.0]
    p['prior']['m.V'] = [0.0, 10.0]
    p['prior']['g.LR_colormix'] = [0.0, 10.0]
    p['prior']['c.LR_colormix'] = [0.0, 10.0]
    p['prior']['a.LR_colormix'] = [0.0, 10.0]
    p['prior']['m.LR_colormix'] = [0.0, 10.0]
    p['prior']['g.S_colormix'] = [0.0, 10.0]
    p['prior']['c.S_colormix'] = [0.0, 10.0]
    p['prior']['a.S_colormix'] = [0.0, 10.0]
    p['prior']['m.S_colormix'] = [0.0, 10.0]
    return p

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

def read_data(cur,conn,params):
    ens_list = params['ens']
    fit_n = params['fit_n']
    op_list = params['op']
    edata = []
    xlist = []
    # read data
    for ens in ens_list:
        # mpi
        sql_cmd = "SELECT (result->>'E0')::double precision FROM callat_proj.meson_v1_bs m JOIN callat_corr.hisq_bootstrap b ON m.bs_id=b.id WHERE meson_v1_id=%s AND mbs = %s AND nbs=0;" %(params['mes_idx'][ens]['mpi'],params['mes_idx'][ens]['mbs'])
        cur.execute(sql_cmd)
        mpi = cur.fetchone()[0]
        # Z0p
        sql_cmd = "SELECT (result->>'Z0_p')::double precision FROM callat_proj.meson_v1_bs m JOIN callat_corr.hisq_bootstrap b ON m.bs_id=b.id WHERE meson_v1_id=%s AND mbs = %s AND nbs=0;" %(params['mes_idx'][ens]['mpi'],params['mes_idx'][ens]['mbs'])
        cur.execute(sql_cmd)
        z0p = cur.fetchone()[0]
        # mres
        sql_cmd = "SELECT (result->>'mres')::double precision FROM callat_proj.mres_v1_bs m JOIN callat_corr.hisq_bootstrap b ON m.bs_id=b.id WHERE mres_v1_id=%s AND mbs = %s AND nbs=0;" %(params['mes_idx'][ens]['mres'],params['mes_idx'][ens]['mbs'])
        cur.execute(sql_cmd)
        mres = cur.fetchone()[0]
        # ml
        sql_cmd = "SELECT mq1::double precision FROM public.proj_summary_meson_v1 WHERE id=%s;" %params['mes_idx'][ens]['mpi']
        cur.execute(sql_cmd)
        ml = cur.fetchone()[0]
        # make fpi
        fpi = z0p*(2.*ml+2.*mres)/mpi**(3./2.)
        eps = mpi/(4.*np.pi*fpi)
        # a/w0
        sql_cmd = "SELECT aw0::double precision FROM callat_corr.hisq_params_bootstrap p JOIN callat_corr.hisq_ensembles e ON p.hisq_ensembles_id=e.id WHERE tag='%s' AND nbs=0;" %ens
        cur.execute(sql_cmd)
        aw0 = cur.fetchone()[0]
        # get matrix element
        otemp = []
        for o in op_list:
            for n in fit_n:
                sql_cmd = "SELECT result::double precision FROM callat_proj.n0bb_v1 WHERE hisq_ensembles='%s' AND operator='%s' and fit_n=%s ORDER BY nbs;" %(ens,o,n)
                cur.execute(sql_cmd)
                temp = cur.fetchall()
                otemp.append(np.array([i[0] for i in temp]))
                xlist.append({'tag':ens,'op':o,'n':n,'eps':eps,'aw0':aw0})
        edata.append(gv.dataset.avg_data(np.array(otemp).T,bstrap=True))
    data = np.array(edata).flatten()
    return xlist, data

def make_priors(params):
    p = dict()
    for k in params['prior'].keys():
        for o in params['op']:
            if k.split('.')[1] == o:
                p[k] = gv.gvar(params['prior'][k][0],params['prior'][k][1])
            else:
                pass
    return p
         

class fit_functions():
    def __init__(self):
        return None
    def unitary(self,x,p):
        fit = []
        for i in x:
            if i['op'] in ['V']:
                fit.append(self.u_V(i,p))
            elif i['op'] in ['LR','LR_colormix','S','S_colormix']:
                fit.append(self.u_LRS(i,p))
        return fit
    def u_V(self,x,p):
        # LO + NLO log
        fit = (4.*np.pi*x['eps'])**2*p['g.V']*(1.-16./3.*x['eps']**2*np.log(x['eps']**2))
        # NLO counterterm
        fit += x['eps']**4*p['c.V']
        # NLO discretization
        fit += x['eps']**2 * x['aw0']**2 * p['a.V']
        return fit
    def u_LRS(self,x,p):
        # LO + NLO log
        fit = p['g.%s' %x['op']]*(1.-10./3.*x['eps']**2*np.log(x['eps']**2))
        # NLO counterterm
        fit += x['eps']**2*p['c.%s' %x['op']]
        # NLO discretization
        fit += x['aw0']**2*p['a.%s' %x['op']]
        return fit

def fit_data(x,y,p):
    prior = make_priors(p)
    fitc = fit_functions()
    fit = lsqfit.nonlinear_fit(data=(x,y),prior=prior,fcn=fitc.unitary,maxit=1000000)
    print fit
    return {'fit': fit, 'prior': prior}

def phys_point(p,result):
    def error_breakdown(fit,prior,r,o):
        e  = dict()
        e['stat']   = r.partialsdev(fit.y)
        e['chiral'] = r.partialsdev(prior['g.%s' %o],prior['c.%s' %o],prior['m.%s' %o])
        e['cont']   = r.partialsdev(prior['a.%s' %o])
        return e
    fit = result['fit']
    prior = result['prior']
    mpi_phys = 139.57018 # MeV
    fpi_phys = 93 # MeV
    eps_phys = mpi_phys/(4.*np.pi*fpi_phys)
    x = []
    for o in p['op']:
        x.append({'op':o,'eps':eps_phys,'aw0': 0})
    fitc = fit_functions()
    result = fitc.unitary(x,fit.p)
    rdict = dict()
    edict = dict()
    string = ''
    for i,o in enumerate(p['op']):
        rdict[o] = result[i]
        edict[o] = error_breakdown(fit,prior,result[i],o)
        string += '%s: %s +/- %s\n' %(o,rdict[o].mean,rdict[o].sdev)
        string += 'stat: %5.3f pct\n' %(np.absolute(edict[o]['stat']/rdict[o].mean*100))
        string += 'chir: %5.3f pct\n' %(np.absolute(edict[o]['chiral']/rdict[o].mean*100))
        string += 'cont: %5.3f pct\n' %(np.absolute(edict[o]['cont']/rdict[o].mean*100))
        string += 'tota: %5.3f pct\n' %(np.absolute(rdict[o].sdev/rdict[o].mean*100))
        string += '\n'
    print string
    return rdict, edict

def plot_data(p,x,y,result):
    def select_alpha(tag):
        if tag in ['l1648f211b580m013m065m838', 'l2448f211b580m0064m0640m828', 'l3248f211b580m00235m0647m831']:
            color = '#c82506' # red
        elif tag in ['l2464f211b600m0102m0509m635', 'l3264f211b600m00507m0507m628', 'l4864f211b600m00184m0507m628', 'l2464f211b600m00507m0507m628', 'l4064f211b600m00507m0507m628','l2464f211b600m0130m0509m635']:
            color = '#70b741' # green
        else:
            color = '#51a7f9' # blue
        return color
    def select_color(operator):
        if operator in ['V']:
            color = '#c82506' # red
        elif operator in ['LR']:
            color = '#70b741' # green
        elif operator in ['LR_colormix']:
            color = '#0b5d12' # dark green
        elif operator in ['S']:
            color = '#51a7f9' # blue
        elif operator in ['S_colormix']:
            color = '#00397a' # dark blue
        return color
    def select_symbol(operator):
        if operator in ['V']:
            symbol = 'o' # o
        elif operator in ['LR']:
            symbol = '^' # triangle up
        elif operator in ['LR_colormix']:
            symbol = 'v' # triangle down
        elif operator in ['S']:
            symbol = 's' # square
        elif operator in ['S_colormix']:
            symbol = 'D' # diamond
        return symbol
    # unpack result 
    fit = result['fit']
    prior = result['prior']
    # plot correlator data
    eps = []
    color = []
    symbol = []
    for i in x:
        eps.append(i['eps'])
        color.append(select_color(i['op']))
        symbol.append(select_symbol(i['op']))
    mean = []
    sdev = []
    for i in y:
        mean.append(i.mean)
        sdev.append(i.sdev)
    fig = plt.figure(figsize=(7,4.326237))
    ax = plt.axes([0.15,0.15,0.8,0.8])
    for i in range(len(eps)):
        ax.errorbar(x=eps[i],y=mean[i],yerr=sdev[i],ls='None',marker=symbol[i],fillstyle='none',markersize='5',elinewidth=1,capsize=2,color=color[i])
    # plot finite a fit
    xlist = []
    for i in x:
        xlist.append(i['aw0'])
    xlist = np.unique(xlist)
    xa = np.linspace(0.0001, 0.2701, 271)
    fitc = fit_functions()
    xal = []
    for o in p['op']:
        for a in xlist:
            xal = []
            for i in range(len(xa)):
                xal.append({'op':o,'eps':xa[i],'aw0':a})
            r = fitc.unitary(xal,fit.p)
            mean = np.array([i.mean for i in r])
            ax.errorbar(x=xa,y=mean,ls='-',marker='None',fillstyle='none',elinewidth=1,color=select_color(o))
    # plot continuum fit
    xc = np.linspace(0.0001, 0.2701, 271)
    xcl = []
    for o in p['op']:
        xcl = []
        for i in range(len(xc)):
            xcl.append({'op':o,'eps':xc[i],'aw0':0})
        r = fitc.unitary(xcl,fit.p)
        mean = np.array([i.mean for i in r])
        sdev = np.array([i.sdev for i in r])
        ax.fill_between(xc,mean+sdev,mean-sdev,alpha=0.4,facecolor=select_color(o)) # purple
    # plot formatting
    ax.set_xlim([0,xc[-1]])
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    ax.set_xlabel('$\epsilon_\pi$', fontsize=20)
    ax.set_ylabel('$\mathcal{O}_i$', fontsize=20)
    plt.draw()
    plt.show()
                
if __name__=="__main__":
    psqlpwd = pwd.passwd()
    cur, conn = login('cchang5','cchang5',psqlpwd)
    p = params()
    x,y = read_data(cur,conn,p)
    result = fit_data(x,y,p)
    phys_point(p,result)
    plot_data(p,x,y,result)
