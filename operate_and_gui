import math, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import gurobipy as gp
from gurobipy import GRB, tuplelist
import pandapower.networks as pn
from scipy.optimize import minimize_scalar
from collections import deque
import networkx as nx
import tkinter as tk
from tkinter import ttk, messagebox

# --------------------
# 전역 파라미터
# --------------------
T = 96                    # 15분 × 96
delta_t = 0.25
vmin, vmax = 0.95, 1.05
soc_min, soc_max = 0.1, 0.9
slack_bus_name = 0
line_lim = 3
kVAbase = 1000
MVAbase = kVAbase / 1000.0
NODE_SIZE = 150           # 네트워크 노드 크기 고정

PEAK_CAP_MW = 2.0
PEN_LSHED   = 1e5
PEN_CURT    = 2e3

# --------------------
# 네트워크 & 전처리
# --------------------
net = pn.case33bw()
impbase = net.bus.vn_kv[0]**2 / MVAbase

frombus = [net.bus.loc[net.line.from_bus[i], 'name'] for i in net.line.index]
tobus   = [net.bus.loc[net.line.to_bus[i], 'name'] for i in net.line.index]
net.line.from_bus = frombus
net.line.to_bus   = tobus
net.line.index    = list(zip(frombus, tobus))
net.line = net.line.loc[net.line.in_service].copy()

net.line['rateA'] = line_lim
net.line['r'] = net.line.length_km * net.line.r_ohm_per_km
net.line['x'] = net.line.length_km * net.line.x_ohm_per_km
net.line['r_pu'] = net.line['r'] / impbase
net.line['x_pu'] = net.line['x'] / impbase
line = net.line[['from_bus','to_bus','r','x','r_pu','x_pu','rateA']]

net.bus.index = net.bus.name
net.bus = net.bus[['name','vn_kv']]
net.bus['Pd'] = 0.0; net.bus['Qd'] = 0.0
for i in net.load.index:
    bname = net.bus.loc[net.load.loc[i,'bus'],'name']
    net.bus.loc[bname,'Pd'] += float(net.load.loc[i,'p_mw'])   * MVAbase
    net.bus.loc[bname,'Qd'] += float(net.load.loc[i,'q_mvar']) * MVAbase
bus = net.bus

# 서브스테이션(ext_grid)
ext = net.ext_grid.copy()
ext.index = [ net.bus.index[b] for b in ext['bus'] ]
ext = ext[['min_p_mw','max_p_mw','min_q_mvar','max_q_mvar']].rename(
    columns={'min_p_mw':'Pmin','max_p_mw':'Pmax','min_q_mvar':'Qmin','max_q_mvar':'Qmax'}
)
ext[['Pmin','Pmax','Qmin','Qmax']] *= MVAbase
ext['b1']=0.0; ext['b2']=0.0; ext['b3']=0.0
ext['gci']=0.30

# DG / PV / ESS
dgloc = {
    'bus':[9,23,30],
    'Pmin':[0,0,0],'Pmax':[0.5,0.5,0.5],
    'Qmin':[-0.3,-0.3,-0.3],'Qmax':[0.3,0.3,0.3],
    'b1':[0.001,0.001,0.001],'b2':[0.005,0.005,0.005],'b3':[0.05,0.05,0.05],
    'gci':[0.40,0.45,0.50]
}
dg  = pd.DataFrame(dgloc).set_index('bus')
gen = pd.concat([ext, dg], axis=0)[['Pmin','Pmax','Qmin','Qmax','b1','b2','b3','gci']]
set_gen = gen.index.tolist()  # [0(slack), 9,23,30]

pvloc = {'bus':[6,16,20],'p_max':[0.3,0.3,0.3],'q_min':[-0.3,-0.3,-0.3],'q_max':[0.3,0.3,0.3]}
pvdata = pd.DataFrame(pvloc).set_index('bus'); set_pv = pvdata.index.tolist()

essloc = {'bus':[7,26],'Cap':[0.8,0.8],'Pmin':[0,0],'Pmax':[0.6,0.6],'Qmin':[-0.1,-0.1],'Qmax':[0.1,0.1]}
essdata = pd.DataFrame(essloc).set_index('bus'); set_ess = essdata.index.tolist()

# --------------------
# 부하 프로파일
# --------------------
try:
    loaddem_raw = np.genfromtxt('LD69-15mins.txt')
    if not np.isfinite(loaddem_raw).all() or len(loaddem_raw) < T:
        raise ValueError("load file invalid")
except Exception:
    loaddem_raw = (np.sin(np.linspace(0, 2*np.pi, T)) + 1.5) * 0.5
l30 = loaddem_raw[:T] / max(1e-9, np.max(loaddem_raw[:T]))

# --------------------
# DRCCP (robust PV)
# --------------------
class DRCCP:
    def __init__(self, data, T, alpha=0.4, N=10, noise_std_fraction=0.1, beta=0.05, scale=1):
        self.data = np.array(data); self.T=T; self.alpha=float(alpha); self.N=N
        self.noise=noise_std_fraction; self.beta=beta; self.scale=scale
    def _normalize(self): self.base = self.data/self.scale
    def _scenarios(self):
        self.scen = np.zeros((self.N,self.T)); mx=max(1e-9,np.max(self.base))
        rng = np.random.default_rng(1234)
        for s in range(self.N):
            for t in range(self.T):
                v=self.base[t]
                self.scen[s,t]=0 if v==0 else np.clip(v+rng.normal(0,self.noise*max(v,1e-4)),0,mx)
        self.Pmax=np.max(self.scen,0); self.Pmin=np.min(self.scen,0)
    def _eps(self):
        log_term=np.log(1/(1-self.beta)); self.eps=[]
        for t in range(self.T):
            if self.base[t]==0: self.eps.append(0.0); continue
            x=self.scen[:,t]; mu=np.mean(x); d2=(x-mu)**2
            def f(rho):
                if rho<=1e-9: return np.inf
                try:
                    me=np.mean(np.exp(rho*d2))
                    if me<=0: return np.inf
                    val=(2.0/rho)*(1+np.log(me))
                    return val if val>=0 else np.inf
                except: return np.inf
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res=minimize_scalar(f,bounds=(1e-7,200),method='bounded')
            D=math.sqrt(res.fun) if res.success and np.isfinite(res.fun) and res.fun>=0 else 0.1
            self.eps.append(D*math.sqrt(log_term/self.N))
    def _solve(self):
        m=gp.Model('DRCCP'); m.Params.OutputFlag=0
        P=m.addVars(self.T,lb=0,ub=1.0); v=m.addVars(self.T,lb=-GRB.INFINITY); z=m.addVars(self.T,self.N,lb=0)
        m.setObjective(gp.quicksum(P[t] for t in range(self.T)), GRB.MAXIMIZE)
        for t in range(self.T):
            m.addConstr(P[t] <= float(self.Pmax[t]))  # 상계
            if self.base[t]>0:
                m.addConstr(self.alpha*self.N*v[t]-gp.quicksum(z[t,s] for s in range(self.N))>=self.eps[t]*self.N)
                for s in range(self.N):
                    m.addConstr(v[t]-z[t,s] <= -P[t] + float(self.scen[s,t]))
            else:
                P[t].ub=0
        m.Params.NonConvex=2; m.optimize()

        out=[0.0]*self.T
        if m.Status==GRB.OPTIMAL:
            for t in range(self.T):
                val=float(P[t].X)
                if self.base[t]>0 and val < 1e-6:  # 0 붕괴 방지: alpha-quantile fallback
                    q = float(np.quantile(self.scen[:,t], min(max(self.alpha,0.0),1.0)))
                    val = max(0.0, min(q, float(self.Pmax[t])))
                out[t]=val
        return out
    def solve(self): self._normalize(); self._scenarios(); self._eps(); return self._solve()

def load_pv_raw_series(bus_id: int):
    """실제(원시) PV series 로드 → [0..1] 정규화"""
    try:
        filemap={6:"jeonnam_PV1.csv",16:"jeonnam_PV2.csv",20:"jeonnam_PV3.csv"}
        df=pd.read_csv(filemap[bus_id], header=None)
        pv_raw=df.iloc[:,0].values
        if len(pv_raw)!=T: raise ValueError("wrong length")
    except Exception:
        pv_raw = np.sin(np.linspace(0,np.pi,T))**2 * 800
    base = pv_raw/max(1e-9,np.max(pv_raw))
    return base

def build_pv_profiles(use_drccp: bool, alpha: float = 0.4):
    pv_profiles={}
    for b in set_pv:
        base = load_pv_raw_series(b)
        pv_profiles[b] = DRCCP(base,T,alpha=alpha).solve() if use_drccp else base
    return pv_profiles

# --------------------
# 선로 유틸
# --------------------
def resi(m,n):
    idx=(m,n) if (m,n) in line.index else (n,m)
    return line.at[idx,'r']/impbase
def reac(m,n):
    idx=(m,n) if (m,n) in line.index else (n,m)
    return line.at[idx,'x']/impbase
def gij1(m,n):
    r=resi(m,n); x=reac(m,n); return r/(r**2+x**2+1e-12)
def gij2(m,n):
    r=resi(m,n); x=reac(m,n); return x/(r**2+x**2+1e-12)

# 인덱스
set_t   = list(range(T))
set_bus = bus.index.tolist()
set_line= line.index.tolist()
bus_t   = tuplelist([(i,t) for t in set_t for i in set_bus])
line_t  = tuplelist([(i,j,t) for t in set_t for i,j in set_line])
line_t_dir = tuplelist([(i,j,t) for t in set_t for i,j in set_line] +
                       [(j,i,t) for t in set_t for i,j in set_line])
gen_t   = tuplelist([(i,t) for t in set_t for i in set_gen])
pv_t    = tuplelist([(i,t) for t in set_t for i in set_pv])
ess_t   = tuplelist([(i,t) for t in set_t for i in set_ess])

# --------------------
# Proposed 해 풀고 시각화용 시계열 수집
# --------------------
def solve_proposed(pv_profiles=None):
    pv_profiles = pv_profiles if pv_profiles is not None else build_pv_profiles(True)

    gen_loc = gen.copy()
    for col in ['b1','b2','b3']:
        gen_loc.loc[gen_loc.index != slack_bus_name, col] *= 1.20
    gen_loc.loc[gen_loc.index != slack_bus_name, 'Pmax'] *= 0.90

    m = gp.Model("Proposed"); m.Params.OutputFlag=0

    theta = m.addVars(bus_t, lb=-math.pi, ub=math.pi, name='Theta')
    v     = m.addVars(bus_t, lb=vmin, ub=vmax, name='V')
    p_g   = m.addVars(gen_t, lb=0, ub=GRB.INFINITY, name='P_Gen')
    q_g   = m.addVars(gen_t, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='Q_Gen')
    p_inj = m.addVars(bus_t, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='P_Inj')
    q_inj = m.addVars(bus_t, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='Q_Inj')
    p_ij  = m.addVars(line_t, lb=[-line.rateA[i] for i in set_line]*T,
                      ub=line.rateA.tolist()*T, name='P_Line')
    q_ij  = m.addVars(line_t, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='Q_Line')

    # 발전기 경계
    m.addConstrs((p_g[i,t] >= gen_loc.Pmin[i] for i,t in gen_t))
    m.addConstrs((p_g[i,t] <= gen_loc.Pmax[i] for i,t in gen_t))
    m.addConstrs((q_g[i,t] >= gen_loc.Qmin[i] for i,t in gen_t))
    m.addConstrs((q_g[i,t] <= gen_loc.Qmax[i] for i,t in gen_t))

    # 선로 방향 + SOS
    p_hat = m.addVars(line_t_dir, lb=0,
                      ub=[line.rateA[i] for i in set_line]*T*2, name='P_hat')
    for t in set_t:
        for i,j in set_line:
            m.addSOS(GRB.SOS_TYPE1, [p_hat[i,j,t], p_hat[j,i,t]],[1,1])
    m.addConstrs((p_ij[i,j,t] == p_hat[i,j,t] - p_hat[j,i,t] for i,j,t in line_t))

    # PV/ESS
    p_pv2grid = m.addVars(pv_t, lb=0, ub=pvdata.p_max.tolist()*T, name='PV2Grid')
    p_pv2ess  = m.addVars(pv_t, lb=0, ub=pvdata.p_max.tolist()*T, name='PV2ESS')
    curt_pv   = m.addVars(pv_t, lb=0, name='PV_Curt')
    q_pv      = m.addVars(pv_t, lb=pvdata.q_min.tolist()*T, ub=pvdata.q_max.tolist()*T, name='Q_PV')

    soc  = m.addVars(ess_t, lb=soc_min, ub=soc_max, name='SOC')
    p_ch = m.addVars(ess_t, lb=0, ub=essdata.Pmax.tolist()*T, name='P_CH')
    p_ch_grid = m.addVars(ess_t, lb=0, ub=essdata.Pmax.tolist()*T, name='P_CH_GRID')
    p_ch_pv   = m.addVars(ess_t, lb=0, ub=essdata.Pmax.tolist()*T, name='P_CH_PV')
    p_dc = m.addVars(ess_t, lb=0, ub=essdata.Pmax.tolist()*T, name='P_DC')
    x_ch = m.addVars(ess_t, vtype=GRB.BINARY, name='X_CH')

    # 전력 흐름 근사
    m.addConstrs((p_ij[i,j,t] ==
                  gij1(i,j)*(theta[i,t]-theta[j,t]) + gij2(i,j)*(v[i,t]-v[j,t]) for i,j,t in line_t))
    m.addConstrs((q_ij[i,j,t] ==
                  -gij2(i,j)*(theta[i,t]-theta[j,t]) + gij1(i,j)*(v[i,t]-v[j,t]) for i,j,t in line_t))
    m.addConstrs((p_ij.sum(i,'*',t) - p_ij.sum('*',i,t) == p_inj[i,t] for i,t in bus_t))
    m.addConstrs((q_ij.sum(i,'*',t) - q_ij.sum('*',i,t) == q_inj[i,t] for i,t in bus_t))
    m.addConstrs((v[slack_bus_name,t] == 1.0 for t in set_t))
    m.addConstrs((theta[slack_bus_name,t] == 0.0 for t in set_t))

    # PV 가능량
    def pvmax(i,t): return pvdata.p_max[i] * (pv_profiles[i][t] if pv_profiles else 1.0)

    # PV 분배
    m.addConstrs((p_pv2grid[i,t] + p_pv2ess[i,t] + curt_pv[i,t] == pvmax(i,t) for i,t in pv_t))
    # PV→ESS 총합 == ESS에서 PV로 충전 총합
    m.addConstrs((gp.quicksum(p_pv2ess[i,t] for i in set_pv) ==
                  gp.quicksum(p_ch_pv[j,t]   for j in set_ess) for t in set_t))

    # ESS
    m.addConstrs((soc[i,t] == soc[i,t-1] + (p_ch[i,t]-p_dc[i,t])*delta_t/essdata.Cap[i] for i,t in ess_t if t>0))
    m.addConstrs((soc[i,0] == 0.5 for i in set_ess))
    m.addConstrs((soc[i,0] == soc[i,T-1] for i in set_ess))
    m.addConstrs((p_ch[i,t] == p_ch_grid[i,t] + p_ch_pv[i,t] for i,t in ess_t))
    m.addConstrs((p_ch[i,t] <= essdata.Pmax[i]*x_ch[i,t] for i,t in ess_t))
    m.addConstrs((p_dc[i,t] <= essdata.Pmax[i]*(1-x_ch[i,t]) for i,t in ess_t))

    # 부하 감축
    lshed_p = m.addVars(bus_t, lb=0, name='P_LoadShed')
    lshed_q = m.addVars(bus_t, lb=0, name='Q_LoadShed')

    # 버스 주입
    m.addConstrs((p_inj[i,t] ==
                  p_g.sum(i,t)
                  + (p_pv2grid[i,t] if i in set_pv else 0)
                  + (p_dc[i,t] if i in set_ess else 0)
                  - (p_ch_grid[i,t] if i in set_ess else 0)
                  - bus.Pd[i]*l30[t]
                  + lshed_p[i,t]
                  for i,t in bus_t))
    m.addConstrs((q_inj[i,t] ==
                  q_g.sum(i,t)
                  + (q_pv.sum(i,t) if i in set_pv else 0)
                  - bus.Qd[i]*l30[t]
                  + lshed_q[i,t]
                  for i,t in bus_t))

    # 피크 제약
    m.addConstrs((p_g[slack_bus_name,t] <= PEAK_CAP_MW for t in set_t))

    # 가격/탄소가격(간단)
    try:  pi_p_hour = np.genfromtxt('wholesale_price-1hour-kw.txt')
    except: pi_p_hour = 50 + 10*np.sin(2*np.pi*np.arange(24)/24)
    try:  pi_c_hour = np.genfromtxt('carbon_price-1hour-kg.txt')
    except: pi_c_hour = 5 + 2*np.cos(2*np.pi*np.arange(24)/24)

    # 목적함수
    a1=a2=a3=1.0; a4=1.0
    dg_buses = [b for b in set_gen if b != slack_bus_name]
    C1 = gp.quicksum((gen_loc.loc[i,'b1']*p_g[i,t]*p_g[i,t] + gen_loc.loc[i,'b2']*p_g[i,t] + gen_loc.loc[i,'b3']) * delta_t
                     for t in set_t for i in dg_buses)
    C2 = gp.quicksum(pi_p_hour[t//4] * p_g[slack_bus_name,t] * delta_t for t in set_t)
    C3 = gp.quicksum(pi_c_hour[t//4] * gp.quicksum(gen_loc.loc[i,'gci']*p_g[i,t] for i in set_gen) * delta_t for t in set_t)
    C4 = gp.quicksum((v[i,t]-vmin) for i,t in bus_t)
    C5 = PEN_LSHED * (gp.quicksum(lshed_p[i,t] for i,t in bus_t) + gp.quicksum(lshed_q[i,t] for i,t in bus_t))
    C6 = PEN_CURT  *  gp.quicksum(curt_pv[i,t] for i,t in pv_t)
    m.setObjective(a1*C1 + a2*C2 + a3*C3 + a4*C4 + C5 + C6, GRB.MINIMIZE)

    m.Params.NonConvex=2; m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"최적화 실패 status={m.Status}")

    # 결과 수집
    gen_series = {i: np.array([p_g[i,t].X for t in set_t]) for i in set_gen}
    sub_t = gen_series[slack_bus_name]
    dg_t  = sum(gen_series[i] for i in set_gen if i!=slack_bus_name)

    soc_mat = np.zeros((T, len(set_ess)))
    for e_idx,i in enumerate(set_ess):
        for t in set_t:
            soc_mat[t,e_idx] = m.getVarByName(f"SOC[{i},{t}]").X

    pv2grid_t = np.zeros(T)
    for i in set_pv:
        for t in set_t:
            pv2grid_t[t] += m.getVarByName(f"PV2Grid[{i},{t}]").X

    pdc_t = np.zeros(T); pch_grid_t = np.zeros(T)
    for i in set_ess:
        for t in set_t:
            pdc_t[t]      += m.getVarByName(f"P_DC[{i},{t}]").X
            pch_grid_t[t] += m.getVarByName(f"P_CH_GRID[{i},{t}]").X

    base_consumer_load = np.zeros(T)
    for t in set_t:
        for i in set_bus:
            base_consumer_load[t] += bus.Pd[i]*l30[t]

    lshed_t = np.zeros(T)
    for t in set_t:
        for i in set_bus:
            lshed_t[t] += m.getVarByName(f"P_LoadShed[{i},{t}]").X

    load_served_t = base_consumer_load - lshed_t
    loss_t = (sub_t + dg_t + pv2grid_t + pdc_t) - (load_served_t + pch_grid_t)
    loss_t = np.maximum(loss_t, 0.0)

    bus_list = list(set_bus)
    pos_idx = {b:k for k,b in enumerate(bus_list)}
    v_mat = np.zeros((len(bus_list), T))
    for (i,t) in bus_t:
        v_mat[pos_idx[i], t] = m.getVarByName(f"V[{i},{t}]").X
    v_mean_t = v_mat.mean(axis=0)

    # 네트워크 흐름/손실/NCI 준비
    Pnet = { (i,j): np.zeros(T) for (i,j) in set_line }
    Qnet = { (i,j): np.zeros(T) for (i,j) in set_line }
    Phat_fwd = { (i,j): np.zeros(T) for (i,j) in set_line }
    Phat_bwd = { (i,j): np.zeros(T) for (i,j) in set_line }
    for (i,j) in set_line:
        for t in set_t:
            Pnet[(i,j)][t] = m.getVarByName(f"P_Line[{i},{j},{t}]").X
            Qnet[(i,j)][t] = m.getVarByName(f"Q_Line[{i},{j},{t}]").X
            Phat_fwd[(i,j)][t] = m.getVarByName(f"P_hat[{i},{j},{t}]").X
            Phat_bwd[(i,j)][t] = m.getVarByName(f"P_hat[{j},{i},{t}]").X

    rpu_map = {(i,j): line.at[(i,j),'r_pu'] if (i,j) in line.index else line.at[(j,i),'r_pu'] for (i,j) in set_line}
    Lline = { (i,j): rpu_map[(i,j)]*(Pnet[(i,j)]**2 + Qnet[(i,j)]**2) for (i,j) in set_line }

    Pinj = { i: np.zeros(T) for i in set_bus }
    Qinj = { i: np.zeros(T) for i in set_bus }
    for i in set_bus:
        for t in set_t:
            Pinj[i][t] = m.getVarByName(f"P_Inj[{i},{t}]").X
            Qinj[i][t] = m.getVarByName(f"Q_Inj[{i},{t}]").X

    # 탄소강도(NCI)
    p_inflow = {t: {i: [] for i in set_bus} for t in set_t}
    for t in set_t:
        for (i,j) in set_line:
            f_ij = Phat_fwd[(i,j)][t]; f_ji = Phat_bwd[(i,j)][t]
            if f_ij > 1e-9: p_inflow[t][j].append((i, f_ij))
            if f_ji > 1e-9: p_inflow[t][i].append((j, f_ji))

    gci_map = gen['gci'].to_dict()
    Pgen_loc = {t:{i:0.0 for i in set_bus} for t in set_t}
    for t in set_t:
        for i in set_gen:
            Pgen_loc[t][i] += gen_series[i][t]

    adj = {i:set() for i in set_bus}
    for (i,j) in set_line:
        adj[i].add(j); adj[j].add(i)
    order = []
    seen=set([slack_bus_name])
    dq=deque([slack_bus_name])
    while dq:
        u=dq.popleft(); order.append(u)
        for v in adj[u]:
            if v not in seen:
                seen.add(v); dq.append(v)

    bus_idx_map = {b:k for k,b in enumerate(set_bus)}
    e_nci = np.zeros((len(set_bus), T))
    for t in set_t:
        e = {i:0.0 for i in set_bus}
        for i in order:
            Pin = Pgen_loc[t][i] + sum(v for _,v in p_inflow[t][i])
            Rin = gci_map.get(i,0.0)*Pgen_loc[t][i] + sum(e[src]*v for src,v in p_inflow[t][i])
            e[i] = (Rin/Pin) if Pin>1e-12 else 0.0
            e_nci[bus_idx_map[i], t] = e[i]

    carbon_edges_t = []
    for t in set_t:
        edges=[]
        for (i,j) in set_line:
            fwd = Phat_fwd[(i,j)][t]; bwd = Phat_bwd[(i,j)][t]
            if fwd > 1e-6:
                edges.append((i,j, float(e_nci[bus_idx_map[i],t]*fwd)))
            if bwd > 1e-6:
                edges.append((j,i, float(e_nci[bus_idx_map[j],t]*bwd)))
        carbon_edges_t.append(edges)

    t_star = int(np.argmax(sub_t))

    time_varying_load_pd = pd.DataFrame(
        np.outer(net.bus['Pd'], l30[:T]),
        index=net.bus.index,
        columns=[f'Time_{t}' for t in range(T)]
    )
    time_varying_load_qd = pd.DataFrame(
        np.outer(net.bus['Qd'], l30[:T]),
        index=net.bus.index,
        columns=[f'Time_{t}' for t in range(T)]
    )

    # --- PV 입지 점수 (Top-10) ---
    # 주간 유입합(예: 08:00~17:00)
    import_day = {i:0.0 for i in set_bus}
    for t in range(16, 68):
        for i in set_bus:
            import_day[i] += sum(v for _,v in p_inflow[t][i])

    # 전기적 거리(경로저항 합)
    parent = {slack_bus_name: None}
    dq2 = deque([slack_bus_name]); seen=set([slack_bus_name])
    while dq2:
        u = dq2.popleft()
        for v in adj[u]:
            if v not in seen:
                seen.add(v); parent[v]=u; dq2.append(v)
    r_map = {(i,j): line.at[(i,j) if (i,j) in line.index else (j,i), 'r_pu'] for (i,j) in set_line}
    pathR = {i:0.0 for i in set_bus}
    for i in set_bus:
        cur=i
        while parent.get(cur) is not None:
            p=parent[cur]
            pathR[i]+= r_map[(cur,p)] if (cur,p) in r_map else r_map[(p,cur)]
            cur=p

    nci_avg = e_nci.mean(axis=1)
    bus_idx = {b:k for k,b in enumerate(set_bus)}
    def zscore_dict(d):
        arr = np.array(list(d.values()), dtype=float)
        mu, sd = arr.mean(), arr.std() if arr.std()>1e-9 else 1.0
        return {k:(v-mu)/sd for k,v in d.items()}
    Zimp = zscore_dict(import_day)
    Zr   = zscore_dict(pathR)
    m_, s_ = nci_avg.mean(), nci_avg.std() if nci_avg.std()>1e-9 else 1.0
    Znci = {i:(nci_avg[bus_idx[i]]-m_)/s_ for i in set_bus}
    pv_score = {i: Zimp[i] - Zr[i] + Znci[i] for i in set_bus}  # 유입↑, 거리↓, NCI↑
    pv_rank  = sorted(pv_score.items(), key=lambda x: x[1], reverse=True)[:10]

    # 레이아웃
    F = nx.Graph(); F.add_edges_from(line.index)
    pos = nx.spring_layout(F, iterations=100, seed=420)

    return dict(
        gen_series=gen_series, substation_series=sub_t, dg_total_series=dg_t,
        soc_mat=soc_mat, pch_grid_t=pch_grid_t, base_consumer_load=base_consumer_load,
        total_grid_load=base_consumer_load + pch_grid_t,
        pv2grid_t=pv2grid_t, pdc_t=pdc_t, lshed_t=lshed_t,
        loss_t=loss_t, v_mean=v_mean_t, v_by_bus=v_mat,
        Pinj=Pinj, Qinj=Qinj, Pnet=Pnet, Qnet=Qnet, Lline=Lline,
        carbon_edges_t=carbon_edges_t, nci_by_bus_time=e_nci,
        pos_graph=pos, t_star=t_star, set_bus=set_bus, set_line=set_line,
        time_varying_load_pd=time_varying_load_pd, time_varying_load_qd=time_varying_load_qd,
        import_day=import_day, nci_avg=nci_avg, pv_rank=pv_rank
    )

# --------------------
# KPI
# --------------------
def kpis_from_result_proposed(r):
    dt = 0.25
    sub = r['substation_series']; dg  = r['dg_total_series']
    pv2g= r['pv2grid_t']; pdc = r['pdc_t']; chg = r['pch_grid_t']
    base= r['base_consumer_load']; ls  = r['lshed_t']; loss= r['loss_t']
    load_served = base - ls
    total_supply = (sub + dg + pv2g + pdc).sum()*dt
    kpi = {
        "Energy_Substation(MWh)": sub.sum()*dt,
        "Energy_DG(MWh)"        : dg.sum()*dt,
        "Energy_PV2Grid(MWh)"   : pv2g.sum()*dt,
        "Energy_ESS_Dis(MWh)"   : pdc.sum()*dt,
        "Energy_ESS_GridChg(MWh)": chg.sum()*dt,
        "Load_Served(MWh)"      : load_served.sum()*dt,
        "Curtail_Total(MWh)"    : 0.0,
        "LoadShed_Total(MWh)"   : ls.sum()*dt,
        "Loss_Total(MWh)"       : loss.sum()*dt,
        "Peak_Substation(MW)"   : float(sub.max()),
        "Peak_Time(idx)"        : int(np.argmax(sub)+1),
        "Mix_Substation(%)"     : 100*(sub.sum()*dt)/(total_supply+1e-9),
        "Mix_DG(%)"             : 100*(dg.sum()*dt)/(total_supply+1e-9),
        "Mix_PV2Grid(%)"        : 100*(pv2g.sum()*dt)/(total_supply+1e-9),
        "Mix_ESS_Dis(%)"        : 100*(pdc.sum()*dt)/(total_supply+1e-9),
    }
    return pd.Series(kpi)

# --------------------
# 연산 실행
# --------------------
pv_profiles = build_pv_profiles(True, alpha=0.4)
res = solve_proposed(pv_profiles)
kpi_ser = kpis_from_result_proposed(res).round(3)

# --------------------
# Tkinter GUI
# --------------------
root = tk.Tk()
root.title("DSO 운영 대시보드 - IEEE33 Proposed")
root.geometry("1320x860")

# 좌측 입력 패널
left = tk.Frame(root, padx=12, pady=12)
left.pack(side=tk.LEFT, fill=tk.Y)

tk.Label(left, text="기본 정보", font=("Malgun Gothic", 12, "bold")).pack(anchor="w", pady=(0,6))
lbls = ["bus 이름", "PV 위치(예: 6,16,20)", "DG 위치(예: 9,23,30)", "ESS 위치(예: 7,26)", "Slack 위치"]
defaults = ["ieee33", "6,16,20", "9,23,30", "7,26", "0"]
entries = {}
for L, D in zip(lbls, defaults):
    tk.Label(left, text=L).pack(anchor="w")
    e = tk.Entry(left, width=22); e.insert(0,D); e.pack(anchor="w", pady=(0,6))
    entries[L]=e

# --- wasstein 섹션 ---
tk.Label(left, text="wasserstein", font=("Malgun Gothic", 12, "bold")).pack(anchor="w", pady=(10,4))
tk.Label(left, text="alpha (0~1)").pack(anchor="w")
alpha_entry = tk.Entry(left, width=10); alpha_entry.insert(0, "0.4")
alpha_entry.pack(anchor="w", pady=(0,6))

def open_pv_uncertainty():
    try:
        pv_list = [int(x.strip()) for x in entries["PV 위치(예: 6,16,20)"].get().split(",") if x.strip()]
        target_bus = pv_list[0]
    except:
        messagebox.showerror("오류", "PV 위치 형식을 확인하세요. 예: 6,16,20")
        return
    base = load_pv_raw_series(target_bus)  # [0..1]
    pmax = float(pvdata.loc[target_bus, 'p_max'])  # MW capacity
    alphas = [round(0.1*i,1) for i in range(5,10)]  # 0.1~0.9
    series = []
    for a in alphas:
        tight = DRCCP(base, T, alpha=a).solve()
        series.append((a, np.array(tight)*pmax))
    fig, ax = plt.subplots(figsize=(10,5), dpi=100)
    ax.plot(np.array(base)*pmax, label="Actual(normalized) × Pmax", linewidth=2)
    for a, s in series: ax.plot(s, label=f"tight α={a:.1f}", alpha=0.95)
    ax.set_title(f"PV Uncertainty (Wasserstein-DRCCP) @ Bus {target_bus}")
    ax.set_xlabel("Time Step (15-min)"); ax.set_ylabel("Power (MW)")
    ax.grid(True); ax.legend(ncol=2, fontsize=9); fig.tight_layout()
    win = tk.Toplevel(root); win.title("태양광 발전 불확실성 (α sweep)")
    canvas = FigureCanvasTkAgg(fig, master=win); canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

tk.Button(left, text="태양광 발전 불확실성", command=open_pv_uncertainty).pack(anchor="w", pady=(4,10))

# 우측 메뉴 패널
right = tk.Frame(root, padx=12, pady=12)
right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

tk.Label(right, text="메뉴", font=("Malgun Gothic", 12, "bold")).pack(anchor="w", pady=(0,6))
btns_frame = tk.Frame(right); btns_frame.pack(anchor="w", pady=(0,6))

def show_fig_in_toplevel(fig, title="Figure"):
    win = tk.Toplevel(root); win.title(title)
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    return win, canvas

# 1) KPI
def open_kpi_window():
    win = tk.Toplevel(root); win.title("KPI 보고서"); win.geometry("540x560")
    cols = ("항목", "값"); tree = ttk.Treeview(win, columns=cols, show="headings", height=22)
    for c in cols:
        tree.heading(c, text=c); tree.column(c, anchor='center', width=260 if c=="항목" else 200)
    for k, v in kpi_ser.to_dict().items():
        tree.insert("", tk.END, values=(k, v))
    style = ttk.Style(win); style.theme_use("default")
    style.configure("Treeview.Heading", font=("Malgun Gothic", 10, "bold"))
    tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# ---- 공통 타임라인 플레이어 (win.after 사용) ----
class TimelinePlayer:
    def __init__(self, win, init_fn, update_fn, title):
        self.win = win
        self.init_fn = init_fn
        self.update_fn = update_fn
        self.playing = True
        self.t = 0
        # figure
        self.fig = plt.Figure(figsize=(10,6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=win)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # controls
        ctl = tk.Frame(win); ctl.pack(fill=tk.X)
        self.btn_play = tk.Button(ctl, text="⏸ 정지", width=8, command=self.toggle)
        self.btn_play.pack(side=tk.LEFT, padx=6, pady=6)
        self.scale = tk.Scale(ctl, from_=0, to=T-1, orient=tk.HORIZONTAL, length=720,
                              label="시간 (t)", command=self.on_seek)
        self.scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        tk.Label(ctl, text="0.1s/frame, loop").pack(side=tk.RIGHT, padx=8)
        # init draw once
        self.art = self.init_fn(self.ax, 0)
        self.fig.suptitle(title); self.canvas.draw()
        # loop
        self.loop()

    def loop(self):
        if self.playing:
            self.t = (self.t + 1) % T
            self.scale.set(self.t)
            self.update_fn(self.ax, self.art, self.t)
            self.canvas.draw_idle()
        self.win.after(100, self.loop)

    def toggle(self):
        self.playing = not self.playing
        self.btn_play.configure(text="▶ 재생" if not self.playing else "⏸ 정지")

    def on_seek(self, val):
        self.t = int(float(val))
        self.update_fn(self.ax, self.art, self.t)
        self.canvas.draw_idle()

# 2) 시계열 네트워크 흐름 보기 (서브메뉴)
def open_timeseries_network_menu():
    menu_win = tk.Toplevel(root)
    menu_win.title("시계열 네트워크 흐름 보기 (2-1 ~ 2-4)")
    menu_win.geometry("360x220")
    tk.Label(menu_win, text="원하는 항목을 선택하세요.", font=("Malgun Gothic", 11, "bold")).pack(pady=10)
    frm = tk.Frame(menu_win); frm.pack(pady=6)

    tk.Button(frm, text="2-1 노드 P/Q & 라인 손실", width=40,
              command=animate_pq_loss).grid(row=0, column=0, padx=6, pady=4)
    tk.Button(frm, text="2-2 노드 전압", width=40,
              command=animate_voltage).grid(row=1, column=0, padx=6, pady=4)
    tk.Button(frm, text="2-3 탄소 흐름", width=40,
              command=animate_carbon_flow).grid(row=2, column=0, padx=6, pady=4)
    tk.Button(frm, text="2-4 노드 탄소 강도(NCI)", width=40,
              command=animate_nci).grid(row=3, column=0, padx=6, pady=4)

# 공통
pos = res['pos_graph']
set_line_local = res['set_line']
set_bus_local  = res['set_bus']

def _draw_graph_base(ax):
    ax.clear(); ax.set_axis_off()

# 2-1: 노드 P/Q, 라인 손실 (간결 라벨)
def animate_pq_loss():
    win = tk.Toplevel(root); win.title("2-1 노드 P/Q & 라인 손실")
    G = nx.DiGraph()
    for (i,j) in set_line_local: G.add_edge(i,j)

    def init(ax, t0):
        _draw_graph_base(ax)
        nx.draw_networkx_edges(G, pos, ax=ax, arrows=False, width=1.3, edge_color="#888")
        nodes_artist = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=NODE_SIZE,
                                              node_color="#f0f0ff", edgecolors="#333")
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_color="#000")
        ax.set_title(f"2-1 Node P/Q & Line Loss | t={t0+1}/96")
        return {"nodes_artist": nodes_artist}

    def update(ax, art, t):
        _draw_graph_base(ax)
        nx.draw_networkx_edges(G, pos, ax=ax, arrows=False, width=1.3, edge_color="#888")
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=NODE_SIZE, node_color="#f0f0ff", edgecolors="#333")
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_color="#000")

        # 노드 라벨: "13.3MW + 3.1Mvar"
        for n in set_bus_local:
            x,y = pos[n]
            p = res['Pinj'][n][t]*1000; q = res['Qinj'][n][t]*1000
            ax.text(x, y+0.06, f"{p:.1f}kW + {q:.1f}kvar", ha='center', fontsize=8, color="#003366")

        # 선로 손실: 빨간 글씨 "0.9MW"
        for (i,j) in set_line_local:
            x=(pos[i][0]+pos[j][0])/2; y=(pos[i][1]+pos[j][1])/2
            loss_mw = res['Lline'][(i,j)][t] * MVAbase*1000
            ax.text(x, y, f"{loss_mw:.1f}kW", ha='center', fontsize=8, color="red")
        ax.set_title(f"2-1 Node P/Q & Line Loss | t={t+1}/96")

    TimelinePlayer(win, init, update, "2-1 Node P/Q & Line Loss")

# 2-2: 전압 (컬러바 1회 생성, facecolor 업데이트)
def animate_voltage():
    win = tk.Toplevel(root); win.title("2-2 노드 전압")
    G = nx.Graph(); G.add_edges_from(set_line_local)
    import matplotlib.colors as mcolors
    norm = mcolors.Normalize(vmin=0.95, vmax=1.05)
    cmap = plt.cm.viridis
    bus_to_idx = {b:i for i,b in enumerate(set_bus_local)}
    nodes = list(set_bus_local)

    def init(ax, t0):
        ax.set_axis_off()
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#666", width=1.3)
        vals0 = [res['v_by_bus'][bus_to_idx[n], t0] for n in nodes]
        colors0 = cmap(norm(vals0))
        nodes_artist = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=NODE_SIZE,
                                              node_color=colors0, edgecolors="#333")
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=9)
        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm); mappable.set_array([])
        cb = ax.figure.colorbar(mappable, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("Voltage (p.u.)")
        ax.set_title(f"2-2 Node Voltages | t={t0+1}/96")
        return {"nodes_artist": nodes_artist}

    def update(ax, art, t):
        vals = [res['v_by_bus'][bus_to_idx[n], t] for n in nodes]
        art["nodes_artist"].set_facecolor(cmap(norm(vals)))
        ax.set_title(f"2-2 Node Voltages | t={t+1}/96")

    TimelinePlayer(win, init, update, "2-2 Node Voltages")

# 2-3: 탄소 흐름 (숫자 라벨 추가)
def animate_carbon_flow():
    win = tk.Toplevel(root); win.title("2-3 탄소 흐름")

    def init(ax, t0):
        _draw_graph_base(ax)
        edges = res['carbon_edges_t'][t0]
        G = nx.DiGraph()
        for (i,j,w) in edges: G.add_edge(i,j,weight=w)
        widths = [max(0.5, min(6.0, 0.8 + 0.6*np.log1p(abs(G[i][j]['weight'])))) for i,j in G.edges()]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=NODE_SIZE, node_color="#ffeeee", edgecolors="#333")
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=9)
        nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, width=widths, edge_color="red", alpha=0.85)
        # 숫자 라벨: 탄소흐름 강도(CF)
        for (i,j) in G.edges():
            x=(pos[i][0]+pos[j][0])/2; y=(pos[i][1]+pos[j][1])/2
            cf = G[i][j]['weight']
            ax.text(x, y, f"CF {cf:.2f}", color="darkred", fontsize=8, ha='center', va='center')
        ax.set_title(f"2-3 Carbon Flow | t={t0+1}/96")
        return {}

    def update(ax, art, t):
        _draw_graph_base(ax)
        edges = res['carbon_edges_t'][t]
        G = nx.DiGraph()
        for (i,j,w) in edges: G.add_edge(i,j,weight=w)
        widths = [max(0.5, min(6.0, 0.8 + 0.6*np.log1p(abs(G[i][j]['weight'])))) for i,j in G.edges()]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=NODE_SIZE, node_color="#ffeeee", edgecolors="#333")
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=9)
        nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, width=widths, edge_color="red", alpha=0.85)
        for (i,j) in G.edges():
            x=(pos[i][0]+pos[j][0])/2; y=(pos[i][1]+pos[j][1])/2
            cf = G[i][j]['weight']
            ax.text(x, y, f"CF {cf:.2f}", color="darkred", fontsize=8, ha='center', va='center')
        ax.set_title(f"2-3 Carbon Flow | t={t+1}/96")

    TimelinePlayer(win, init, update, "2-3 Carbon Flow")

# 2-4: NCI (컬러바 1회 생성, facecolor 업데이트)
def animate_nci():
    win = tk.Toplevel(root); win.title("2-4 노드 탄소 강도")
    e_nci = res['nci_by_bus_time']
    G = nx.Graph(); G.add_edges_from(set_line_local)
    import matplotlib.colors as mcolors
    vmax = max(1e-9, float(np.max(e_nci)))
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.cm.plasma
    bus_to_idx = {b:i for i,b in enumerate(set_bus_local)}
    nodes = list(set_bus_local)

    def init(ax, t0):
        ax.set_axis_off()
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#666", width=1.3)
        vals0 = [e_nci[bus_to_idx[n], t0] for n in nodes]
        colors0 = cmap(norm(vals0))
        nodes_artist = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=NODE_SIZE,
                                              node_color=colors0, edgecolors="#333")
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=9)
        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm); mappable.set_array([])
        cb = ax.figure.colorbar(mappable, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("NCI (relative)")
        ax.set_title(f"2-4 Node Carbon Intensity (NCI) | t={t0+1}/96")
        return {"nodes_artist": nodes_artist}

    def update(ax, art, t):
        vals = [e_nci[bus_to_idx[n], t] for n in nodes]
        art["nodes_artist"].set_facecolor(cmap(norm(vals)))
        ax.set_title(f"2-4 Node Carbon Intensity (NCI) | t={t+1}/96")

    TimelinePlayer(win, init, update, "2-4 Node Carbon Intensity (NCI)")

# 3) 발전기 출력
def open_gen_outputs():
    gen_order = [slack_bus_name] + [i for i in set_gen if i!=slack_bus_name]
    labels = ['slack'] + [f"DG {i}" for i in gen_order[1:]]
    pp = np.column_stack([res['gen_series'][i] for i in gen_order])
    fig, ax = plt.subplots(figsize=(10,5), dpi=100)
    for idx, lab in enumerate(labels): ax.plot(pp[:, idx], label=lab)
    ax.legend(title='Generator'); ax.set_xlabel('Time Step (15-min)'); ax.set_ylabel('Power (MW)')
    ax.set_title('Generator Outputs (Proposed)'); ax.grid(True); fig.tight_layout()
    show_fig_in_toplevel(fig, "발전기 작동 현황")

# 4) ESS SOC
def open_ess_soc():
    fig, ax = plt.subplots(figsize=(10,5), dpi=100)
    for e_idx,i in enumerate(set_ess): ax.plot(res['soc_mat'][:, e_idx], label=f'ESS@Bus {i}')
    ax.legend(); ax.set_xlabel('Time Step (15-min)'); ax.set_ylabel('SOC (p.u.)')
    ax.set_title('ESS State of Charge (Proposed)'); ax.grid(True); fig.tight_layout()
    show_fig_in_toplevel(fig, "ESS 작동 상태")

# 5) System Load Breakdown
def open_system_load():
    load_df = pd.DataFrame({
        'Base_Consumer_Load': res['base_consumer_load'],
        'ESS_Charging_Load' : res['pch_grid_t'],
        'Total_Grid_Load'   : res['total_grid_load'],
    })
    fig, ax = plt.subplots(figsize=(12,5), dpi=100)
    ax.plot(load_df['Base_Consumer_Load'], label='Base Consumer Load', linestyle='--')
    ax.plot(load_df['ESS_Charging_Load'], label='ESS Charging (from Grid)', linestyle=':')
    ax.plot(load_df['Total_Grid_Load'], label='Total Grid Load', linewidth=2, marker='.')
    ax.set_title('System Load Breakdown per Time Step (Proposed)')
    ax.set_xlabel('Time Step (15-min)'); ax.set_ylabel('Active Power (kW)')
    ax.legend(); ax.grid(True); fig.tight_layout()
    show_fig_in_toplevel(fig, "시스템 부하 상태")

# 6) PV 입지 추천 Top-10
def open_pv_siting():
    pv_rank = res.get('pv_rank', [])
    if not pv_rank:
        messagebox.showwarning("알림", "PV 점수 데이터가 없습니다.")
        return
    labels = [str(b) for b,_ in pv_rank]
    scores = [s for _,s in pv_rank]
    fig, ax = plt.subplots(figsize=(8,4), dpi=100)
    ax.bar(labels, scores)
    ax.set_title("PV Siting Score (Top-10)")
    ax.set_xlabel("Bus"); ax.set_ylabel("Score (normalized sum)")
    fig.tight_layout()
    show_fig_in_toplevel(fig, "PV 입지 추천 (Top-10)")

# 오른쪽 버튼들
btns = [
    tk.Button(btns_frame, text="1. KPI 보고서", width=30, command=open_kpi_window),
    tk.Button(btns_frame, text="2. 계통 네트워크", width=30, command=open_timeseries_network_menu),
    tk.Button(btns_frame, text="3. 발전 작동 현황", width=30, command=open_gen_outputs),
    tk.Button(btns_frame, text="4. ESS 작동 상태", width=30, command=open_ess_soc),
    tk.Button(btns_frame, text="5. 시스템 부하 상태", width=30, command=open_system_load),
    tk.Button(btns_frame, text="6. PV 입지 Top-10", width=30, command=open_pv_siting),
]
for i, b in enumerate(btns):
    b.grid(row=i//2, column=i%2, padx=8, pady=6, sticky="w")

tk.Label(right, text="※ 애니메이션: 슬라이더로 임의 시점 이동, ▶/⏸로 제어 (0.1s/frame)", fg="#555").pack(anchor="w", pady=(8,0))

root.mainloop()
