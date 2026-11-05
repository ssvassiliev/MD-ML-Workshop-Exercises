import h5py
import numpy as np
import parmed as pmd
from parmed.amber import NetCDFTraj

npt_steps=200
timestep_ps=0.001

def hdf5_to_nc(h5_file, nc_file, natom, dt_frame_ps, stride=1, t0_ps=0.0):
    traj = NetCDFTraj.open_new(nc_file, natom, box=True, crds=True, vels=False)
    with h5py.File(h5_file, 'r') as f:
        pos, box = f['positions'], f['box']          # (k, batch, N, 3), (k, batch, 1)
        k, batch = pos.shape[:2]
        i = 0
        for ik in range(k):                          # stream one flush at a time
            P = pos[ik].astype(np.float64)
            B = box[ik].astype(np.float64)
            for ib in range(batch):
                if i % stride == 0:
                    a = float(B[ib, 0])
                    traj.add_time(t0_ps + i * dt_frame_ps)
                    traj.add_coordinates(P[ib])
                    traj.add_box([a, a, a, 90., 90., 90.])
                i += 1
    traj.close()

# The HDF5 has no virtual sites, so the topology must not have them either.
struct = pmd.load_file('4/polyAla_4.prmtop')
struct.strip('@EPW')                                 # same atoms your `keep` mask removed
struct.save('4/polyAla_4_novsite.prmtop', overwrite=True)

hdf5_to_nc('hdf5/trajectory.hdf5', 'trajectory.nc',
           natom=len(struct.atoms), dt_frame_ps=npt_steps * timestep_ps)
