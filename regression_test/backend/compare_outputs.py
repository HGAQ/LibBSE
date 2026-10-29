"""Compare TDA eigenpairs modulo phase and rotations within degenerate groups."""
from pathlib import Path
import ast
import pprint
import tomllib
import argparse,re
import numpy as np
RY_EV=13.605693122994

def read_eigenpairs(output_root):
    excitation_energies=np.loadtxt(output_root/'Excitation_Energy_singlet.dat',ndmin=1)*RY_EV
    amplitude_paths=list(output_root.glob('Excitation_Amplitude_singlet_*.dat'))
    amplitude_ranks={int(path.stem.rsplit('_',1)[1]):path for path in amplitude_paths}
    if not amplitude_ranks or sorted(amplitude_ranks)!=list(range(len(amplitude_ranks))):
        raise ValueError('Missing or noncontiguous amplitude ranks')
    amplitude_parts=[]
    for rank in range(len(amplitude_ranks)):
        amplitude_rows=[]
        for amplitude_line in amplitude_ranks[rank].read_text().splitlines():
            if not amplitude_line.strip():
                continue
            amplitude_tokens=re.findall(r'\(([^,]+),([^\)]+)\)',amplitude_line)
            amplitude_rows.append([complex(float(real),float(imag)) for real,imag in amplitude_tokens])
        amplitude_part=np.asarray(amplitude_rows,dtype=complex)
        if amplitude_part.ndim!=2 or len(amplitude_part)!=len(excitation_energies):
            raise ValueError('Amplitude state count does not match energies')
        amplitude_parts.append(amplitude_part)
    excitation_amplitudes=np.concatenate(amplitude_parts,axis=1).T
    if not np.isfinite(excitation_amplitudes).all() or not np.isfinite(excitation_energies).all():
        raise ValueError('Nonfinite eigenpair')
    return excitation_energies,excitation_amplitudes

def compare_eigenpairs(reference_energies,reference_amplitudes,candidate_energies,candidate_amplitudes,
                       energy_atol=2e-6,amplitude_atol=2e-4,degeneracy_ev=1e-5):
    if reference_energies.shape!=candidate_energies.shape or reference_amplitudes.shape!=candidate_amplitudes.shape:
        raise ValueError('Eigenpair dimensions differ')
    if np.any(np.diff(reference_energies)<-energy_atol) or np.any(np.diff(candidate_energies)<-energy_atol):
        raise ValueError('Eigenvalues must be sorted')
    max_energy_error=float(np.max(np.abs(reference_energies-candidate_energies)))
    reference_norms=np.linalg.norm(reference_amplitudes,axis=0)
    candidate_norms=np.linalg.norm(candidate_amplitudes,axis=0)
    normalization_error=float(max(np.max(abs(reference_norms-1)),np.max(abs(candidate_norms-1))))
    if min(reference_norms.min(),candidate_norms.min())<.5:
        raise ValueError('Invalid eigenvector norms')
    orthogonality_error=float(max(np.max(np.abs(reference_amplitudes.conj().T@reference_amplitudes-np.eye(len(reference_energies)))),np.max(np.abs(candidate_amplitudes.conj().T@candidate_amplitudes-np.eye(len(candidate_energies))))))
    max_subspace_error=0.
    max_aligned_error=0.
    group_count=0
    state_matches=[]
    group_begin=0
    for group_end in range(1,len(reference_energies)+1):
        if group_end<len(reference_energies) and reference_energies[group_end]-reference_energies[group_begin]<=degeneracy_ev:
            continue
        reference_group=reference_amplitudes[:,group_begin:group_end]
        candidate_group=candidate_amplitudes[:,group_begin:group_end]
        reference_basis,_=np.linalg.qr(reference_group)
        candidate_basis,_=np.linalg.qr(candidate_group)
        overlap_matrix=reference_basis.conj().T@candidate_basis
        overlap_singular=np.linalg.svd(overlap_matrix,compute_uv=False)
        subspace_error=float(np.sqrt(max(0.,1.-min(1.,float(overlap_singular.min()))**2)))
        max_subspace_error=max(max_subspace_error,subspace_error)
        if group_end-group_begin==1:
            overlap_value=np.vdot(reference_group[:,0],candidate_group[:,0])
            phase_factor=overlap_value.conjugate()/abs(overlap_value) if abs(overlap_value)>0 else 1.
            max_aligned_error=max(max_aligned_error,float(np.linalg.norm(candidate_group[:,0]*phase_factor-reference_group[:,0])))
        # Singleton overlap and degenerate subspace fidelity share this invariant.
        match_squared=min(1.,float(overlap_singular.min())**2)
        for state_index in range(group_begin,group_end):
            state_matches.append({
                'state':state_index+1,
                'reference_energy_eV':float(reference_energies[state_index]),
                'candidate_energy_eV':float(candidate_energies[state_index]),
                'energy_difference_eV':float(candidate_energies[state_index]-reference_energies[state_index]),
                'group_first':group_begin+1,'group_last':group_end,
                'match_squared':match_squared,'subspace_sine':subspace_error})
        group_count+=1
        group_begin=group_end
    return {'pass':max_energy_error<=energy_atol and max_subspace_error<=amplitude_atol and max_aligned_error<=amplitude_atol and normalization_error<=amplitude_atol and orthogonality_error<=amplitude_atol,
        'max_energy_error_eV':max_energy_error,'max_subspace_sine':max_subspace_error,
        'max_phase_aligned_amplitude_l2':max_aligned_error,'max_normalization_error':normalization_error,
        'max_orthogonality_error':orthogonality_error,'groups':group_count,'energy_atol_eV':energy_atol,'amplitude_atol':amplitude_atol,'degeneracy_eV':degeneracy_ev,'states':state_matches}

def read_screening(output_root):
    screening_records={}
    for screening_path in output_root.glob('screening_q_*.bin'):
        with screening_path.open('rb') as handle:
            auxiliary_count=np.fromfile(handle,np.int32,1)
            if len(auxiliary_count)!=1: raise ValueError('Missing screening dimensions')
            auxiliary_count=int(auxiliary_count[0])
            screening_header=np.fromfile(handle,float,4)
            screening_sqrt=np.fromfile(handle,complex,auxiliary_count**2).reshape(auxiliary_count,auxiliary_count)
            screening_inverse=np.fromfile(handle,complex,auxiliary_count**2).reshape(auxiliary_count,auxiliary_count)
            if handle.read(1): raise ValueError('Trailing screening bytes')
        screening_key=tuple(np.round(np.r_[np.mod(screening_header[:3]+1e-10,1)-1e-10,screening_header[3]],9))
        if screening_key in screening_records: raise ValueError('Duplicate screening q/frequency')
        screening_records[screening_key]=(screening_sqrt,screening_inverse)
    return screening_records

def compare_screening(reference_screening,candidate_screening):
    if reference_screening.keys()!=candidate_screening.keys(): raise ValueError('Screening q/frequency coverage differs')
    screening_metrics=[]
    for screening_key in reference_screening:
        reference_sqrt,reference_inverse=reference_screening[screening_key]
        candidate_sqrt,candidate_inverse=candidate_screening[screening_key]
        reference_w=reference_sqrt@reference_inverse@reference_sqrt
        candidate_w=candidate_sqrt@candidate_inverse@candidate_sqrt
        if not np.isfinite(candidate_w).all(): raise ValueError('Nonfinite W')
        relative_error=float(np.linalg.norm(candidate_w-reference_w)/max(np.linalg.norm(reference_w),1e-30))
        screening_metrics.append({'q_frequency':[float(component) for component in screening_key],'relative_W_error':relative_error})
    return {'pass':all(item['relative_W_error']<=2e-8 for item in screening_metrics),'q_count':len(screening_metrics),'matrices':screening_metrics,'relative_tolerance':2e-8}

def compare_directories(reference_root,candidate_root,require_screening=False):
    reference_energies,reference_amplitudes=read_eigenpairs(reference_root)
    candidate_energies,candidate_amplitudes=read_eigenpairs(candidate_root)
    comparison_metrics={'eigenpairs':compare_eigenpairs(reference_energies,reference_amplitudes,candidate_energies,candidate_amplitudes)}
    reference_screening=read_screening(reference_root)
    candidate_screening=read_screening(candidate_root)
    if require_screening and not reference_screening: raise ValueError('Required W diagnostics missing')
    comparison_metrics['screening']=compare_screening(reference_screening,candidate_screening)
    comparison_metrics['pass']=comparison_metrics['eigenpairs']['pass'] and comparison_metrics['screening']['pass']
    return comparison_metrics

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('reference',type=Path)
    parser.add_argument('candidate',type=Path)
    parser.add_argument('--require-screening',action='store_true')
    args=parser.parse_args()
    comparison_metrics=compare_directories(args.reference,args.candidate,args.require_screening)
    print(pprint.pformat(comparison_metrics))
    raise SystemExit(0 if comparison_metrics['pass'] else 1)
