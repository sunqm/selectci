#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Selected CI

Simple usage::

    >>> from pyscf import gto, scf, ao2mo, fci
    >>> mol = gto.M(atom='C 0 0 0; C 0 0 1')
    >>> mf = scf.RHF(mol).run()
    >>> h1 = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)
    >>> h2 = ao2mo.kernel(mol, mf.mo_coeff)
    >>> e = fci.select_ci.kernel(h1, h2, mf.mo_coeff.shape[1], mol.nelectron)[0]
'''

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf.fci import cistring
from pyscf.fci import direct_spin1

def contract_2e(h2e, civec, norb, nelec, hdiag=None, **kwargs):
    strs = civec._strs
    ts = civec._ts
    ndet = len(strs)
    if hdiag is None:
        hdiag = make_hdiag(numpy.zeros((norb,norb)), h2e, strs, norb, nelec)
    ci1 = numpy.zeros_like(civec)

    h2e = h2e.reshape([norb]*4)
    neleca, nelecb = nelec
    vja = numpy.einsum('iipq->pq', h2e[:neleca,:neleca])
    vjb = numpy.einsum('iipq->pq', h2e[:nelecb,:nelecb])
    vka = numpy.einsum('piiq->pq', h2e[:,:neleca,:neleca])
    vkb = numpy.einsum('piiq->pq', h2e[:,:nelecb,:nelecb])
    focka = vja+vjb - vka
    fockb = vja+vjb - vkb
    H = numpy.diag(hdiag)

    for ip in range(ndet):
        for jp in range(ip):
            if abs(ts[ip] - ts[jp]).sum() > 2:
                continue
            stria, strib = strs[ip].reshape(2,-1)
            strja, strjb = strs[jp].reshape(2,-1)
            desa, crea = str_diff(stria, strja)
            if len(desa) > 2:
                continue
            desb, creb = str_diff(strib, strjb)
            if len(desb) + len(desa) > 2:
                continue

            if len(desa) + len(desb) == 1:
# alpha->alpha
                if len(desb) == 0:
                    i,a = desa[0], crea[0]
                    occsa = str2orblst(stria, norb)[0]
                    occsb = str2orblst(strib, norb)[0]
                    fai = 0
                    for k in occsa:
                        fai += h2e[k,k,a,i] - h2e[k,i,a,k]
                    for k in occsb:
                        fai += h2e[k,k,a,i]
                    sign = cre_des_sign(a, i, stria)
                    ci1[jp] += sign * fai * civec[ip]
                    ci1[ip] += sign * fai * civec[jp]
# beta ->beta
                elif len(desa) == 0:
                    i,a = desb[0], creb[0]
                    occsa = str2orblst(stria, norb)[0]
                    occsb = str2orblst(strib, norb)[0]
                    fai = 0
                    for k in occsb:
                        fai += h2e[k,k,a,i] - h2e[k,i,a,k]
                    for k in occsa:
                        fai += h2e[k,k,a,i]
                    sign = cre_des_sign(a, i, strib)
                    ci1[jp] += sign * fai * civec[ip]
                    ci1[ip] += sign * fai * civec[jp]

            else:
# alpha,alpha->alpha,alpha
                if len(desb) == 0:
                    i,j = desa
                    a,b = crea
# 6 conditions for i,j,a,b
# --++, ++--, -+-+, +-+-, -++-, +--+ 
                    if a > j or i > b:
                        v = h2e[a,j,b,i]-h2e[a,i,b,j]
                        sign = cre_des_sign(b, i, stria)
                        sign*= cre_des_sign(a, j, stria)
                    else:
                        v = h2e[a,i,b,j]-h2e[a,j,b,i]
                        sign = cre_des_sign(b, j, stria)
                        sign*= cre_des_sign(a, i, stria)
                    ci1[jp] += sign * v * civec[ip]
                    ci1[ip] += sign * v * civec[jp]
# beta ,beta ->beta ,beta
                elif len(desa) == 0:
                    i,j = desb
                    a,b = creb
                    if a > j or i > b:
                        v = h2e[a,j,b,i]-h2e[a,i,b,j]
                        sign = cre_des_sign(b, i, strib)
                        sign*= cre_des_sign(a, j, strib)
                    else:
                        v = h2e[a,i,b,j]-h2e[a,j,b,i]
                        sign = cre_des_sign(b, j, strib)
                        sign*= cre_des_sign(a, i, strib)
                    ci1[jp] += sign * v * civec[ip]
                    ci1[ip] += sign * v * civec[jp]
# alpha,beta ->alpha,beta
                else:
                    i,a = desa[0], crea[0]
                    j,b = desb[0], creb[0]
                    v = h2e[a,i,b,j]
                    sign = cre_des_sign(a, i, stria)
                    sign*= cre_des_sign(b, j, strib)
                    ci1[jp] += sign * v * civec[ip]
                    ci1[ip] += sign * v * civec[jp]
        ci1[ip] += hdiag[ip] * civec[ip]

    return ci1

def make_hdiag(h1e, eri, strs, norb, nelec):
    eri = ao2mo.restore(1, eri, norb)
    diagj = numpy.einsum('iijj->ij',eri)
    diagk = numpy.einsum('ijji->ij',eri)

    ndet = len(strs)
    hdiag = numpy.zeros(ndet)
    for idet, (stra, strb) in enumerate(strs.reshape(ndet,2,-1)):
        aocc = str2orblst(stra, norb)[0]
        bocc = str2orblst(strb, norb)[0]
        e1 = h1e[aocc,aocc].sum() + h1e[bocc,bocc].sum()
        e2 = diagj[aocc][:,aocc].sum() + diagj[aocc][:,bocc].sum() \
           + diagj[bocc][:,aocc].sum() + diagj[bocc][:,bocc].sum() \
           - diagk[aocc][:,aocc].sum() - diagk[bocc][:,bocc].sum()
        hdiag[idet] = e1 + e2*.5
    return hdiag

def cre_des_sign(p, q, string):
    nset = len(string)
    pg, pb = p//64, p%64
    qg, qb = q//64, q%64

    if pg > qg:
        n1 = 0
        for i in range(nset-pg, nset-qg-1):
            n1 += bin(string[i]).count('1')
        n1 += bin(string[-1-pg] & ((1<<pb) - 1)).count('1')
        n1 += string[-1-qg] >> (qb+1)
    elif pg < qg:
        n1 = 0
        for i in range(nset-qg, nset-pg-1):
            n1 += bin(string[i]).count('1')
        n1 += bin(string[-1-qg] & ((1<<qb) - 1)).count('1')
        n1 += string[-1-pg] >> (pb+1)
    else:
        if p > q:
            mask = (1 << pb) - (1 << (qb+1))
        else:
            mask = (1 << qb) - (1 << (pb+1))
        n1 = bin(string&mask).count('1')

    if n1 % 2:
        return -1
    else:
        return 1

def argunique(strs):
    def order(x, y):
        for i in range(y.size):
            if x[i] > y[i]:
                return 1
            elif y[i] > x[i]:
                return -1
        return 0
    def qsort_idx(idx):
        nstrs = len(idx)
        if nstrs <= 1:
            return idx
        else:
            ref = idx[-1]
            group_lt = []
            group_gt = []
            for i in idx[:-1]:
                c = order(strs[i], strs[ref])
                if c == -1:
                    group_lt.append(i)
                elif c == 1:
                    group_gt.append(i)
            return qsort_idx(group_lt) + [ref] + qsort_idx(group_gt)
    return qsort_idx(range(len(strs)))

def argunique_with_t(strs, ts):
    if len(strs) == 0:
        return []
    else:
        strs = numpy.asarray(strs)
        ts = numpy.asarray(ts, dtype=numpy.int32)
        t_ab = ts.view(numpy.int64).ravel()
        idxs = []
        for ti in numpy.unique(t_ab):
            idx = numpy.where(t_ab == ti)[0]
            idxs.append(idx[argunique(strs[idx])])
        return numpy.hstack(idxs)

def str_diff(string0, string1):
    des_string0 = []
    cre_string0 = []
    nset = len(string0)
    off = 0
    for i in reversed(range(nset)):
        df = string0[i] ^ string1[i]
        des_string0.extend([x+off for x in find1(df & string0)])
        cre_string0.extend([x+off for x in find1(df & string1)])
        off += 64
    return des_string0, cre_string0

def excitation_level(string, nelec=None):
    nset = len(string)
    if nelec is None:
        nelec = 0
        for i in range(nset):
            nelec += bin(string[i]).count('1')

    g, b = nelec//64, nelec%64
    tn = nelec - bin(string[-1-g])[64-b:].count('1')
    for s in string[nset-g:]:
        tn -= bin(s).count('1')
    return tn

def find1(s):
    return [i for i,x in enumerate(bin(s)[2:][::-1]) if x is '1']

def toggle_bit(s, place):
    nset = len(s)
    g, b = place//64, place%64
    s[-1-g] ^= (1<<b)
    return s

def select_strs(myci, civec, h1, eri, norb, nelec):
    eri = eri.reshape([norb]*4)
    neleca, nelecb = nelec
    vja = numpy.einsum('iipq->pq', eri[:neleca,:neleca])
    vjb = numpy.einsum('iipq->pq', eri[:nelecb,:nelecb])
    vka = numpy.einsum('piiq->pq', eri[:,:neleca,:neleca])
    vkb = numpy.einsum('piiq->pq', eri[:,:nelecb,:nelecb])
    focka = h1 + vja+vjb - vka
    fockb = h1 + vja+vjb - vkb

    eri = eri.ravel()
    eri_sorted = abs(eri).argsort()[::-1]
    jk = eri.reshape([norb]*4)
    jk = jk - jk.transpose(2,1,0,3)
    jkf = jk.ravel()
    jk_sorted = abs(jkf).argsort()[::-1]

    ndet = len(civec._strs)
    str_add = []
    t_add = []
    for idet, (stra, strb) in enumerate(civec._strs.reshape(ndet,2,-1)):
        occsa, virsa = str2orblst(stra, norb)
        occsb, virsb = str2orblst(strb, norb)
        tol = myci.select_cutoff / abs(civec[idet])

# alpha->alpha
        holes = [i for i in virsa if i < neleca]
        particles = [i for i in occsa if i >= neleca]
        for i in occsa:
            for a in virsa:
                fai = focka[a,i]
                for k in particles:
                    fai += jk[k,k,a,i]
                for k in holes:
                    fai -= jk[k,k,a,i]
                if abs(fai) > tol:
                    newa = toggle_bit(toggle_bit(stra.copy(), a), i)
                    str_add.append(numpy.hstack((newa,strb)))
                    t_add.append((1,0))
# beta ->beta
        holes = [i for i in virsb if i < nelecb]
        particles = [i for i in occsb if i >= nelecb]
        for i in occsb:
            for a in virsb:
                fai = fockb[a,i]
                for k in particles:
                    fai += jk[k,k,a,i]
                for k in holes:
                    fai -= jk[k,k,a,i]
                if abs(fai) > tol:
                    newb = toggle_bit(toggle_bit(strb.copy(), a), i)
                    str_add.append(numpy.hstack((stra,newb)))
                    t_add.append((0,1))

        for ih in jk_sorted:
            if abs(jkf[ih]) < tol:
                break
            ij, lp = ih//norb, ih%norb
            ij, kp = ij//norb, ij%norb
            ip, jp = ij//norb, ij%norb
# alpha,alpha->alpha,alpha
            if jp in occsa and ip in virsa and lp in occsa and kp in virsa:
                newa = toggle_bit(toggle_bit(toggle_bit(toggle_bit(stra.copy(), jp), ip), lp), kp)
                str_add.append(numpy.hstack((newa,strb)))
                t_add.append((2,0))
# beta ,beta ->beta ,beta
            if jp in occsb and ip in virsb and lp in occsb and kp in virsb:
                newb = toggle_bit(toggle_bit(toggle_bit(toggle_bit(strb.copy(), jp), ip), lp), kp)
                str_add.append(numpy.hstack((stra,newb)))
                t_add.append((0,2))

        for ih in eri_sorted:
            if abs(eri[ih]) < tol:
                break
            ij, lp = ih//norb, ih%norb
            ij, kp = ij//norb, ij%norb
            ip, jp = ij//norb, ij%norb
# alpha,beta ->alpha,beta
            if jp in occsa and ip in virsa and lp in occsb and kp in virsb:
                newa = toggle_bit(toggle_bit(stra.copy(), jp), ip)
                newb = toggle_bit(toggle_bit(strb.copy(), lp), kp)
                str_add.append(numpy.hstack((newa,newb)))
                t_add.append((1,1))

    str_add = numpy.asarray(str_add)
    t_add = numpy.asarray(t_add)
    idx = argunique_with_t(str_add, t_add)
    return str_add[idx], t_add[idx]

def enlarge_space(myci, civec, h1, eri, norb, nelec):
    cidx = abs(civec) > myci.ci_coeff_cutoff
    ci_coeff = civec[cidx]
    strs = civec._strs[cidx]
    ts = civec._ts[cidx]
    str_add, t_add = select_strs(myci, ci_coeff, h1, eri, norb, nelec)

    def largereq(x, y):
        for i in range(y.size):
            if x[i] > y[i]:
                return True
            elif y[i] > x[i]:
                return False
        return True
    def argmerge(strs1, strs2):
        strs = []
        cidx = []
        ndet1 = len(strs1)
        ndet2 = len(strs2)
        p1 = 0
        p2 = 0
        while p1 < ndet1 and p2 < ndet2:
            if largereq(strs1[p1], strs2[p1]):
                strs.append(strs1[p1])
                cidx.append(p2+p1)
                p1 += 1
            else:
                strs.append(strs2[p1])
                ci.append(0)
                p2 += 1
        if p1 < ndet1:
            strs.extend(strs1[p1:])
            cidx.extend(range(p2+p1, p2+ndet1))
        if p2 < ndet2:
            strs.extend(strs2[p2:])
        strs = numpy.asarray(strs)
        cidx = numpy.asarray(cidx)
        return strs, cidx

    ts1 = numpy.asarray(ts, dtype=numpy.int32).view(numpy.int64).ravel()
    ts2 = numpy.asarray(t_add, dtype=numpy.int32).view(numpy.int64).ravel()
    t_ab = numpy.hstack((ts1, ts2))
    uniq_t = numpy.unique(t_ab)
    new_strs = []
    new_ci = []
    new_ts = []
    for ti in uniq_t:
        idx1 = numpy.where(ts1 == ti)[0]
        if len(idx1) > 0:
            strs1 = strs[idx1]
            strs2 = str_add[ts2 == ti]
            s, cidx = argmerge(strs1, strs2)
            new_strs.append(s)
            c = numpy.zeros(len(s))
            c[cidx] = ci_coeff[idx1]
        else:
            strs2 = str_add[ts2 == ti]
            new_strs.append(strs2)
            c = numpy.zeros(len(strs2))
        new_ci.append(c)
        new_ts.append(numpy.repeat(ti, len(c)))

    new_strs = numpy.vstack(new_strs)
    new_ci = numpy.hstack(new_ci)
    new_ts = numpy.hstack(new_ts).view(numpy.int32).reshape(-1,2)
    return _as_SCIvector(new_ci, new_strs, new_ts)

def str2orblst(string, norb):
    occ = []
    vir = []
    nset = len(string)
    off = 0
    for k in reversed(range(nset)):
        s = string[k]
        occ.extend([x+off for x in find1(s)])
        for i in range(0, min(64, norb-off)):
            if not (s & (1<<i)):
                vir.append(i+off)
        off += 64
    return occ, vir

def orblst2str(lst, norb):
    nset = (norb+63) // 64
    string = numpy.zeros(nset, dtype=numpy.int64)
    for i in lst:
        toggle_bit(string, i)
    return string


def to_fci(civec, norb, nelec):
    assert(norb <= 64)
    neleca, nelecb = nelec
    strsa = cistring.gen_strings4orblist(range(norb), neleca)
    stradic = dict(zip(strsa,range(strsa.__len__())))
    strsb = cistring.gen_strings4orblist(range(norb), nelecb)
    strbdic = dict(zip(strsb,range(strsb.__len__())))
    na = len(stradic)
    nb = len(strbdic)
    ndet = len(civec)
    fcivec = numpy.zeros((na,nb))
    for idet, (stra, strb) in enumerate(civec._strs.reshape(ndet,2,-1)):
        ka = stradic[stra[0]]
        kb = strbdic[strb[0]]
        fcivec[ka,kb] = civec[idet]
    return fcivec

def from_fci(fcivec, ci_strs, norb, nelec):
    neleca, nelecb = nelec
    strsa = cistring.gen_strings4orblist(range(norb), neleca)
    stradic = dict(zip(strsa,range(strsa.__len__())))
    strsb = cistring.gen_strings4orblist(range(norb), nelecb)
    strbdic = dict(zip(strsb,range(strsb.__len__())))
    na = len(stradic)
    nb = len(strbdic)
    fcivec = fcivec.reshape(na,nb)
    ta = [excitation_level(s, neleca) for s in strsa.reshape(-1,1)]
    tb = [excitation_level(s, nelecb) for s in strsb.reshape(-1,1)]
    ndet = len(ci_strs)
    civec = numpy.zeros(ndet)
    ts = numpy.zeros((ndet,2), dtype=numpy.int32)
    for idet, (stra, strb) in enumerate(ci_strs.reshape(ndet,2,-1)):
        ka = stradic[stra[0]]
        kb = strbdic[strb[0]]
        civec[idet] = fcivec[ka,kb]
        ts[idet,0] = ta[ka]
        ts[idet,1] = tb[kb]
    return _as_SCIvector(civec, ci_strs, ts)


class SelectedCI(direct_spin1.FCISolver):
    def __init__(self, mol=None):
        direct_spin1.FCISolver.__init__(self, mol)
        self.ci_coeff_cutoff = .5e-3
        self.select_cutoff = .5e-3
        self.conv_tol = 1e-9

##################################################
# don't modify the following attributes, they are not input options
        #self.converged = False
        #self.ci = None
        self._strs = None
        self._keys = set(self.__dict__.keys())


class _SCIvector(numpy.ndarray):
    def __array_finalize__(self, obj):
        self._ts = getattr(obj, '_ts', None)
        self._strs = getattr(obj, '_strs', None)

def _as_SCIvector(civec, ci_strs, ci_ts):
    civec = civec.view(_SCIvector)
    civec._strs = ci_strs
    civec._ts = ci_ts
    return civec

def _as_SCIvector_if_not(civec, ci_strs, ci_ts):
    if not hasattr(civec, '_strs'):
        civec = _as_SCIvector(civec, ci_strs, ci_ts)
    return civec

def _unpack(civec, nelec, ci_strs=None, spin=None):
    neleca, nelecb = direct_spin1._unpack_nelec(nelec, spin)
    ci_strs = getattr(civec, '_strs', ci_strs)
    if ci_strs is not None:
        strsa, strsb = ci_strs
        strsa = numpy.asarray(strsa)
        strsb = numpy.asarray(strsb)
        ci_strs = (strsa, strsb)
    return civec, (neleca, nelecb), ci_strs


if __name__ == '__main__':
    numpy.random.seed(3)
    strs = (numpy.random.random((14,3)) * 4).astype(numpy.uint64)
    print strs
    print argunique(strs)
    ts = numpy.ones((len(strs),2), dtype=int)
    ts[10:,1] = 2
    print argunique_with_t(strs, ts)

    norb = 8
    nelec = 4,4
    hf_str = numpy.hstack([orblst2str(range(nelec[0]), norb),
                           orblst2str(range(nelec[1]), norb)]).reshape(1,-1)
    numpy.random.seed(3)
    h1 = numpy.random.random([norb]*2)**4 * 1e-2
    h1 = h1 + h1.T
    h2 = numpy.random.random([norb]*4)**4 * 1e-2
    h2 = h2 + h2.transpose(0,1,3,2)
    h2 = h2 + h2.transpose(1,0,2,3)
    h2 = h2 + h2.transpose(2,3,0,1)
    ts = numpy.zeros((1,2), dtype=numpy.int32)
    ci1 = _as_SCIvector(numpy.ones(1), hf_str, ts)

    myci = SelectedCI()
    myci.select_cutoff = .001
    myci.ci_coeff_cutoff = .001

    ci2 = enlarge_space(myci, ci1, h1, h2, norb, nelec)
    print len(ci2)
    print numpy.unique(ci2._ts.view(numpy.int64)).view(numpy.int32).reshape(-1,2)

    ci2 = enlarge_space(myci, ci1, h1, h2, norb, nelec)
    numpy.random.seed(1)
    ci2[:] = numpy.random.random(ci2.size)
    ci2 *= 1./numpy.linalg.norm(ci2)
    ci3 = contract_2e(h2, ci2, norb, nelec)

    fci2 = to_fci(ci2, norb, nelec)
    g2e = direct_spin1.absorb_h1e(numpy.zeros_like(h1), h2, norb, nelec, .5)
    fci3 = direct_spin1.contract_2e(g2e, fci2, norb, nelec)
    fci3 = from_fci(fci3, ci2._strs, norb, nelec)

#    H = direct_spin1.pspace(numpy.zeros_like(h1), h2, norb, nelec)[1]
#    neleca, nelecb = nelec
#    strsa = cistring.gen_strings4orblist(range(norb), neleca)
#    stradic = dict(zip(strsa,range(strsa.__len__())))
#    strsb = cistring.gen_strings4orblist(range(norb), nelecb)
#    strbdic = dict(zip(strsb,range(strsb.__len__())))
#    nb = len(strbdic)
#    ndet = len(ci2)
#    idx = []
#    for idet, (stra, strb) in enumerate(ci2._strs.reshape(ndet,2,-1)):
#        ka = stradic[stra[0]]
#        kb = strbdic[strb[0]]
#        idx.append(ka*nb+kb)
#    H = H[idx][:,idx]

    print abs(ci3-fci3).sum()
