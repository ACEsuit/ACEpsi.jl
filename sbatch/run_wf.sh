set -euo pipefail

METHOD="WSSR"                
MOL="Be"                     
CHECKPOINT_PREFIX="checkpoints_" 

NORM_CONSTRAIN="0.01"
ETA="0.95"
DAMPING="0.001"
SVD_RANK="800"
SVD_ITER="3"
LR_DC="2000"
LR="0.0015"

ITER_HEAD="1000"
ITER_TAIL="50000"
BURNIN="5000"
NCHAINS="2048"
LAG="10"
DT="0.08"
ALPHA="1.0"

PROJECT_DIR="/home/dexuan1/projects/rrg-ortner/dexuan1/wf.jl"
JULIA_SCRIPT_DIR="${PROJECT_DIR}/jl"
RESULTS_ROOT="results"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method)          METHOD="${2}"; shift 2 ;;
    --mol)             MOL="${2}"; shift 2 ;;
    --norm|--nc)       NORM_CONSTRAIN="${2}"; shift 2 ;;
    --eta)             ETA="${2}"; shift 2 ;;
    --damping|--d)     DAMPING="${2}"; shift 2 ;;
    --svd-rank|--r)    SVD_RANK="${2}"; shift 2 ;;
    --svd-iter|--si)   SVD_ITER="${2}"; shift 2 ;;
    --lr-dc)           LR_DC="${2}"; shift 2 ;;
    --lr)              LR="${2}"; shift 2 ;;
    --iter-head)       ITER_HEAD="${2}"; shift 2 ;;
    --iter-tail)       ITER_TAIL="${2}"; shift 2 ;;
    --burnin)          BURNIN="${2}"; shift 2 ;;
    --nchains)         NCHAINS="${2}"; shift 2 ;;
    --lag)             LAG="${2}"; shift 2 ;;
    --dt)              DT="${2}"; shift 2 ;;
    --alpha)           ALPHA="${2}"; shift 2 ;;
    --project-dir)     PROJECT_DIR="${2}"; JULIA_SCRIPT_DIR="${PROJECT_DIR}/jl"; shift 2 ;;
    --results-root)    RESULTS_ROOT="${2}"; shift 2 ;;
    --checkpoint-prefix) CHECKPOINT_PREFIX="${2}"; shift 2 ;;
    --help|-h)
      echo "sbatch run_wf.sh [--method SPRING|WSSR|SKETCH] [--mol Be] [--lr 0.0015] ..."
      exit 0
      ;;
    *)
      echo "Unknown parameter: $1"; exit 1 ;;
  esac
done

module purge
module load julia/1.11.3
module load python/3.11

mkdir -p "${SCRATCH}/.julia" logs
export JULIA_DEPOT_PATH="${SCRATCH}/.julia"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export JULIA_PKG_PRECOMPILE_AUTO=0

TAG="${METHOD}/lr=${LR}_lr_dc=${LR_DC}_n=${NORM_CONSTRAIN}_eta=${ETA}_d=${DAMPING}_alpha=${ALPHA}"
if [[ "${METHOD}" == "WSSR" || "${METHOD}" == "SKETCH" ]]; then
  TAG="${TAG}_r=${SVD_RANK}_i=${SVD_ITER}"
fi
RES_PATH="${RESULTS_ROOT}/${MOL}/${TAG}/"

echo "METHOD=${METHOD}"
echo "MOL=${MOL}"
echo "LR=${LR}, LR_DC=${LR_DC}, NORM_CONSTRAIN=${NORM_CONSTRAIN}, ETA=${ETA}, DAMPING=${DAMPING}"
echo "SVD_RANK=${SVD_RANK}, SVD_ITER=${SVD_ITER}"
echo "ITER_HEAD=${ITER_HEAD}, ITER_TAIL=${ITER_TAIL}, BURNIN=${BURNIN}, NCHAINS=${NCHAINS}, LAG=${LAG}, DT=${DT}, ALPHA=${ALPHA}"
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "RES_PATH=${RES_PATH}"
echo
cd "${PROJECT_DIR}"
julia --project=. -e "using Pkg; Pkg.activate(\"${PROJECT_DIR}\"); Pkg.status()"

export WF_METHOD="${METHOD}"
export WF_MOL="${MOL}"
export WF_CHECKPOINT_FILE="${CHECKPOINT_PREFIX}${MOL}.jld2"
export WF_LR="${LR}"
export WF_LR_DC="${LR_DC}"
export WF_NORM="${NORM_CONSTRAIN}"
export WF_ETA="${ETA}"
export WF_DAMPING="${DAMPING}"
export WF_SVD_RANK="${SVD_RANK}"
export WF_SVD_ITER="${SVD_ITER}"
export WF_ITER_HEAD="${ITER_HEAD}"
export WF_ITER_TAIL="${ITER_TAIL}"
export WF_BURNIN="${BURNIN}"
export WF_NCHAINS="${NCHAINS}"
export WF_LAG="${LAG}"
export WF_DT="${DT}"
export WF_ALPHA="${ALPHA}"
export WF_RES_PATH="${RES_PATH}"

julia --project=. <<'JULIA_EOF'
using Pkg
Pkg.activate(ENV["PWD"])
using ACEpsi
using JLD2
using Optimisers: destructure

# ---- ENV ----
method   = get(ENV, "WF_METHOD", "WSSR")
mol_name = get(ENV, "WF_MOL", "Be")
chkfile  = get(ENV, "WF_CHECKPOINT_FILE", "checkpoints_Be.jld2")

lr       = parse(Float64, get(ENV, "WF_LR", "0.0015"))
lr_dc    = parse(Int,    get(ENV, "WF_LR_DC", "2000"))
norm_c   = parse(Float64, get(ENV, "WF_NORM", "0.01"))
eta      = parse(Float64, get(ENV, "WF_ETA", "0.95"))
damping  = parse(Float64, get(ENV, "WF_DAMPING", "0.001"))
svd_rank = parse(Int,    get(ENV, "WF_SVD_RANK", "800"))
svd_iter = parse(Int,    get(ENV, "WF_SVD_ITER", "3"))

iter_head = parse(Int, get(ENV, "WF_ITER_HEAD", "1000"))
iter_tail = parse(Int, get(ENV, "WF_ITER_TAIL", "50000"))
burnin    = parse(Int, get(ENV, "WF_BURNIN", "5000"))
nchains   = parse(Int, get(ENV, "WF_NCHAINS", "2048"))
lag       = parse(Int, get(ENV, "WF_LAG", "10"))
Δt        = parse(Float64, get(ENV, "WF_DT", "0.2"))
alpha     = parse(Float64, get(ENV, "WF_ALPHA", "1.0"))

res_path = get(ENV, "WF_RES_PATH", "results/" * mol_name * "/")

# ---- System ----
atom_name = [mol_name]
atom_sys = mol_name == "Be"  ? [ACEpsi.molecules.Be]  :
           mol_name == "Li2" ? [ACEpsi.molecules.Li2] :
           mol_name == "LiH" ? [ACEpsi.molecules.LiH] :
           mol_name == "N"   ? [ACEpsi.molecules.N]   :
           mol_name == "Ne"  ? [ACEpsi.molecules.Ne]  :
           error("Unknown atom/mol: $mol_name")

for (name_, atom) in zip(atom_name, atom_sys)
    filename = chkfile
    @info "Loading checkpoint" filename
    @load filename p_cpu

    mol = atom
    ν = 2
    family = Plain
    freeze_branches = true

    println("""
    ==== Model setup summary ====
    System:           $name_
    Family:           $family
    Freeze branches:  $freeze_branches
    =============================
    """)

    model = model_generator(
        mol, ν;
        hf         = true,
        ratio      = 1.0,
        multilevel = true,
        family     = family,
        freeze_branches = freeze_branches,
        spec1p_admissible = nothing
    )
    model_list, ps_list, st_list, spec_list, spec1p_list, totdeg_list, ν_list = model
    p, s = destructure(ps_list[1])
    ps_list[1] = s(p_cpu)

    iterations = fill(iter_head, length(model_list))
    iterations[end] = iter_tail

    methods = (
        SPRINGSolver(),
        MINSRSolver(),
        DirectSolver(),
        SketchSolver(800, 50, 50, 1.4),
        SVDSolver(svd_rank, 50, 50, 1.4, svd_iter)
    )

    # SPRING -> methods[1]
    # WSSR   -> methods[5]
    # SKETCH -> methods[4]
    which =
        method == "SPRING" ? 1 :
        method == "WSSR"   ? 5 :
        method == "SKETCH" ? 4 :
        error("Unknown METHOD=$(method). Support SPRING / WSSR / SKETCH")

    solver = methods[which]

    optimizer = OPTSETTING(
        solver;
        iterations = iterations,
        burnin     = burnin,
        nchains    = nchains,
        lag        = lag,
        Δt         = Δt,
        lr         = lr,
        lr_dc      = lr_dc,
        norm_constrain = norm_c,
        η          = eta,
        damping    = 0.01,
        damping_decay = 100,
        damping_min   = damping,
        res_path   = res_path
    )

    @info "Training start" method=method lr=lr lr_dc=lr_dc norm_c=norm_c η=eta damping=damping res_path=res_path
    x0, model_list, ps_list, st_list, val_list, var_list, rank_list =
        train(mol, model_list, ps_list, st_list, spec_list, spec1p_list, totdeg_list, ν_list, optimizer; device=:gpu)

    @info "Done."
end
JULIA_EOF
