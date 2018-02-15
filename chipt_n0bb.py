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
import yaml
import tables
plt.rc('text', usetex=True)

if True:
    plt.rcParams['text.latex.preamble'] = [
        r'\usepackage{helvet}',
        r'\usepackage{sansmath}',
        r'\sansmath']

ms = '3'
cs = 3
fs_l = 7
fs_xy = 7
ts = 7
lw = 0.5
plt_axes = [0.14,0.155,0.825,0.825]

def params():
    # set up data set
    p = dict()
    #p['ens'] = ['l1648f211b580m013m065m838','l2448f211b580m0064m0640m828','l2464f211b600m00507m0507m628','l2464f211b600m0102m0509m635','l3248f211b580m00235m0647m831','l3264f211b600m00507m0507m628','l3296f211b630m0074m037m440','l4064f211b600m00507m0507m628','l4864f211b600m00184m0507m628','l4896f211b630m00363m0363m430']
    p['ens'] = ['l1648f211b580m013m065m838','l2448f211b580m0064m0640m828','l2464f211b600m00507m0507m628','l2464f211b600m0102m0509m635','l3264f211b600m00507m0507m628','l3296f211b630m0074m037m440','l4064f211b600m00507m0507m628','l4864f211b600m00184m0507m628','l4896f211b630m00363m0363m430']
    p['fit_n'] = [1]
    #p['op'] = ['LR','S','V','LR_colormix','S_colormix']
    p['op'] = ['V','LR','LR_colormix','S','S_colormix']
    #p['op'] = ['S']
    #p['op'] = ['V']
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
    # read gV indices
    gVidx = yaml.load(open('./params.yml'))
    p['gVidx'] = dict(gVidx['ensembles'])
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

def ens_translate(ens):
    if ens in ['l1648f211b580m013m065m838','l2448f211b580m0064m0640m828','l3248f211b580m00235m0647m831']:
        r = 'a15'
    elif ens in ['l2464f211b600m0102m0509m635','l2464f211b600m00507m0507m628','l3264f211b600m00507m0507m628','l4064f211b600m00507m0507m628','l4864f211b600m00184m0507m628']:
        r = 'a12'
    elif ens in ['l3296f211b630m0074m037m440','l4896f211b630m00363m0363m430','l6496f211b630m0012m0363m432']:
        r = 'a09'
    return r

def read_data(cur,conn,params):
    ens_list = params['ens']
    fit_n = params['fit_n']
    op_list = params['op']
    edata = []
    xlist = []
    # read data
    for ens in ens_list:
        # a/w0
        sql_cmd = "SELECT aw0::double precision FROM callat_corr.hisq_params_bootstrap p JOIN callat_corr.hisq_ensembles e ON p.hisq_ensembles_id=e.id WHERE tag='%s' AND nbs=0;" %ens
        cur.execute(sql_cmd)
        aw0 = cur.fetchone()[0]
        # get gV
        sql_cmd = "SELECT (result->>'gV00')::double precision FROM callat_proj.ga_v1_bs gv JOIN callat_corr.hisq_bootstrap bs ON gv.bs_id=bs.id WHERE ga_v1_id=%s AND nbs=0;" %(params['gVidx'][ens]['idx']['fh'])
        cur.execute(sql_cmd)
        gV = cur.fetchone()[0]
        # get matrix element
        otemp = [] # temp ME
        etemp = [] # temp epi
        for n in fit_n:
            # get epi
            sql_cmd = "SELECT result::double precision FROM callat_proj.n0bb_v1 WHERE hisq_ensembles='%s' AND operator='epi' and fit_n=%s ORDER BY nbs;" %(ens,n)
            cur.execute(sql_cmd)
            temp = cur.fetchall()
            etemp.append(np.array([i[0] for i in temp]))
            for o in op_list:
                # get renorm coefficients
                # get ME
                sql_cmd = "SELECT result::double precision FROM callat_proj.n0bb_v1 WHERE hisq_ensembles='%s' AND operator='%s' and fit_n=%s ORDER BY nbs;" %(ens,o,n)
                cur.execute(sql_cmd)
                temp = cur.fetchall()
                otemp.append(np.array([i[0]/(aw0**4) for i in temp])) # convert matrix element to w0 units
                xlist.append({'tag':ens,'op':o,'n':n,'aw0':aw0,'gV':gV})
        # apply renorm here on otemp
        # get mq
        sql_cmd = "SELECT mval FROM callat_proj.project_ga_v1_boot0 WHERE id=%s;" %(params['gVidx'][ens]['idx']['fh'])
        cur.execute(sql_cmd)
        mq = cur.fetchone()[0].rstrip('0')
        # get zmat
        if ens in ['l2464f211b600m00507m0507m628', 'l4064f211b600m00507m0507m628']:
            ens = 'l3264f211b600m00507m0507m628'
        else: pass
        if ens == 'l3296f211b630m0074m037m440':
            stream = 'e'
        else: stream = 'a'
        ### renorm coeff from henry from hdf5
        #zcoeff = tables.open_file('./interpolatedZ_V2.hdf5') 
        #z4q = zcoeff.get_node('/fourQuark') 
        #es = ens+stream
        #zmat = z4q.read_where("(ensemble=='%s') & (fixVarValue=='%s')" %(es,mq))[0][1]
        ### renorm coeff from david text files
        dzmat = dict()
        # step scale all
        dzmat['a15'] = gv.gvar([['0.9407(63)','0(0)','0(0)','0(0)','0(0)'],['0(0)','0.9835(68)','-0.0106(18)','0(0)','0(0)'],['0(0)','-0.0369(30)','1.0519(80)','0(0)','0(0)'],['0(0)','0(0)','0(0)','1.020(11)','-0.0356(49)'],['0(0)','0(0)','0(0)','-0.0485(31)','0.9519(68)']])
        dzmat['a12'] = gv.gvar([['0.9117(43)','0(0)','0(0)','0(0)','0(0)'],['0(0)','0.9535(48)','-0.0130(17)','0(0)','0(0)'],['0(0)','-0.0284(30)','0.9922(60)','0(0)','0(0)'],['0(0)','0(0)','0(0)','0.9656(96)','-0.0275(49)'],['0(0)','0(0)','0(0)','-0.0360(28)','0.9270(50)']])
        dzmat['a09'] = gv.gvar([['0.9017(39)','0(0)','0(0)','0(0)','0(0)'],['0(0)','0.9483(44)','-0.0269(17)','0(0)','0(0)'],['0(0)','-0.0236(29)','0.9369(54)','0(0)','0(0)'],['0(0)','0(0)','0(0)','0.9209(91)','-0.0224(49)'],['0(0)','0(0)','0(0)','-0.0230(28)','0.9332(47)']])
        zmat = dzmat[ens_translate(ens)]
        # renormalize data
        dat = {'y': np.array(otemp).T, 'epi': np.array(etemp).T}
        gvdat = gv.dataset.avg_data(dat,bstrap=True)
        gvdat['y'] = zmat.dot(gvdat['y'])/gV**2
        edata.append(gvdat)
    y = []
    epi = dict()
    for idx,i in enumerate(edata):
        ens = ens_list[idx]
        y.append(i['y'])
        epi[ens] = i['epi'][0]
    data = {'y': np.array(y).flatten(), 'epi': epi}
    return xlist, data


def make_priors(y,params):
    p = dict()
    for k in params['prior'].keys():
        for o in params['op']:
            if k.split('.')[1] == o:
                p[k] = gv.gvar(params['prior'][k][0],params['prior'][k][1])
            else:
                pass
    for k in y['epi'].keys():
        p['epi_%s' %k] = y['epi'][k]
    return p
         

class fit_functions():
    def __init__(self):
        return None
    def unitary(self,x,p):
        fit = []
        for i in x:
            k = i['tag']
            epi = p['epi_%s' %k]
            if i['op'] in ['V']:
                fit.append(self.u_V(i,p,epi))
            elif i['op'] in ['LR','LR_colormix','S','S_colormix']:
                fit.append(self.u_LRS(i,p,epi))
        return fit
    def u_V(self,x,p,epi):
        # LO + NLO log
        fit = (4.*np.pi*epi)**2*p['g.V']*(1.-16./3.*epi**2*np.log(epi**2))
        # NLO counterterm
        fit += epi**4*p['c.V']
        # NLO discretization
        fit += epi**2 * x['aw0']**2 * p['a.V']
        return fit
    def u_LRS(self,x,p,epi):
        # LO + NLO log
        fit = p['g.%s' %x['op']]*(1.-10./3.*epi**2*np.log(epi**2))
        # NLO counterterm
        fit += epi**2*p['c.%s' %x['op']]
        # NLO discretization
        fit += x['aw0']**2*p['a.%s' %x['op']]
        return fit

def fit_data(x,y,p):
    prior = make_priors(y,p)
    fitc = fit_functions()
    fit = lsqfit.nonlinear_fit(data=(x,y['y']),prior=prior,fcn=fitc.unitary,maxit=1000000)
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
        x.append({'op':o,'tag': 'phys', 'aw0': 0})
    fitc = fit_functions()
    post = dict()
    for k in prior.keys():
        post[k] = fit.p[k]
    post['epi_phys'] = eps_phys
    result = fitc.unitary(x,post)
    rdict = dict()
    edict = dict()
    string = ''
    inv_w0 = 0.197327/gv.gvar(0.1714,0.0015) #[GeV/fm * fm] MILC gradient flow paper abstract
    for i,o in enumerate(p['op']):
        rdict[o] = result[i] * inv_w0**4
        edict[o] = error_breakdown(fit,prior,rdict[o],o)
        string += '%s: %s +/- %s\n' %(o,rdict[o].mean,rdict[o].sdev)
        #string += 'stat: %5.3f pct\n' %(np.absolute(edict[o]['stat']/rdict[o].mean*100))
        #string += 'chir: %5.3f pct\n' %(np.absolute(edict[o]['chiral']/rdict[o].mean*100))
        #string += 'cont: %5.3f pct\n' %(np.absolute(edict[o]['cont']/rdict[o].mean*100))
        #string += 'tota: %5.3f pct\n' %(np.absolute(rdict[o].sdev/rdict[o].mean*100))
        #string += '\n'
    print string
    return rdict, edict

def plot_data(p,x,y,result):
    def select_alpha(tag):
        if tag in ['l1648f211b580m013m065m838', 'l2448f211b580m0064m0640m828', 'l3248f211b580m00235m0647m831',0.8804]:
            alpha = 1.0
        elif tag in ['l2464f211b600m0102m0509m635', 'l3264f211b600m00507m0507m628', 'l4864f211b600m00184m0507m628', 'l2464f211b600m00507m0507m628', 'l4064f211b600m00507m0507m628','l2464f211b600m0130m0509m635',0.7036]:
            alpha = 0.7
        else:
            alpha = 0.4
        return alpha
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
    alpha = []
    for i in x:
        eps.append(fit.p['epi_%s' %i['tag']])
        color.append(select_color(i['op']))
        symbol.append(select_symbol(i['op']))
        alpha.append(select_alpha(i['tag']))
    mean = []
    sdev = []
    inv_w0 = 0.197327/gv.gvar(0.1714,0.0015) #[GeV/fm * fm] MILC gradient flow paper abstract
    for i in y['y']:
        res = i*inv_w0**4
        mean.append(res.mean)
        sdev.append(res.sdev)
    fig = plt.figure(figsize=(3.50394,2.1655535534))
    ax = plt.axes(plt_axes)
    for i in range(len(eps)):
        ax.errorbar(x=eps[i].mean,y=mean[i],yerr=sdev[i],ls='None',marker=symbol[i],markersize=ms,elinewidth=lw,capsize=cs,mew=lw,fillstyle='none',color=color[i],mfc='black',alpha=alpha[i])
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
                xal.append({'op':o, 'tag':'extrap','aw0':a})
            post = dict()
            for k in prior.keys():
                post[k] = fit.p[k]
            post['epi_extrap'] = xa
            r = fitc.unitary(xal,post)[0]*inv_w0**4
            mean = np.array([i.mean for i in r])
            ax.errorbar(x=xa,y=mean,ls='-',marker='None',fillstyle='none',elinewidth=lw,color=select_color(o),alpha=select_alpha(a))
    # plot continuum fit
    xc = np.linspace(0.0001, 0.2701, 271)
    xcl = []
    for o in p['op']:
        xcl = []
        for i in range(len(xc)):
            xcl.append({'op':o,'tag':'extrap','aw0':0})
        post = dict()
        for k in prior.keys():
            post[k] = fit.p[k]
        post['epi_extrap'] = xc
        r = fitc.unitary(xcl,post)[0]*inv_w0**4
        mean = np.array([i.mean for i in r])
        sdev = np.array([i.sdev for i in r])
        ax.fill_between(xc,mean+sdev,mean-sdev,alpha=0.2,facecolor=select_color(o)) # purple
    # plot formatting
    # for Full Plot
    if True:
        ax.set_xlim([0,xc[-1]])
        #ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.xaxis.set_tick_params(labelsize=ts,width=lw)
        ax.yaxis.set_tick_params(labelsize=ts,width=lw)
        ax.set_xlabel('$\epsilon_\pi$', fontsize=fs_xy)
        ax.set_ylabel('$\mathcal{O}_i$~[GeV${}^4$]', fontsize=fs_xy)
        [i.set_linewidth(lw) for i in ax.spines.itervalues()]
        plt.draw()
        fig.savefig('./n0bb_chipt.pdf',transparent=True)
        plt.show()
    # for V
    else:
        ax.set_xlim([0,xc[-1]])
        ax.set_ylim([-0.0015,0.0005])
        #ax.set_yticks([-0.03, -0.02, -0.01, 0.0, 0.01])
        #ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.xaxis.set_tick_params(labelsize=ts,width=lw)
        ax.yaxis.set_tick_params(labelsize=ts,width=lw)
        ax.set_xlabel('$\epsilon_\pi$', fontsize=fs_xy)
        ax.set_ylabel('$\mathcal{O}_V$~[GeV${}^4$]', fontsize=fs_xy)
        [i.set_linewidth(lw) for i in ax.spines.itervalues()]
        plt.draw()
        fig.savefig('./n0bbO3_chipt.pdf',transparent=True)
        plt.show()
                
if __name__=="__main__":
    psqlpwd = pwd.passwd()
    cur, conn = login('cchang5','cchang5',psqlpwd)
    p = params()
    x,y = read_data(cur,conn,p)
    result = fit_data(x,y,p)
    phys_point(p,result)
    plot_data(p,x,y,result)
