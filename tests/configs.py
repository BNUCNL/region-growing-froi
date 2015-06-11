__author__ = 'zgf'

ACTIVATION_DATA_DIR = "G:/workingdir/BAA/seven/session/combined/all_session.nii.gz"
SUBJECT_ID_DIR = "G:/workingdir/BAA/seven/session/subject_id"
SUBJECTS_ACTIVITION_PRE_DIR = "G:/workingdir/BAA/seven/nfs/t2/fmricenter/obj_reliability/func/"
SUBJECTS_ACTIVITION_SUFFIX_DIR = "/obj.gfeat/cope1.feat/stats/zstat1.nii.gz"
SUBJECTS_LABELS_DIR = "G:/workingdir/BAA/seven/session/t2.3_prob/labels/all_subject_labels_t2.3.nii.gz"
ATLAS_SUBJECTS_LABELS_DIR = "G:/nfs/4Ddata/all_sub_labels.nii.gz"
ALL_202_SUBJECTS_DATA_DIR = "G:/nfs/4Ddata/2006zstat.nii.gz"

RESULT_NPY_FILE = "peak_points_all_sub.npy"
RESULT_CSV_FILE = "peak_points_all_sub.csv"

ROI = ['r_OFA', 'l_OFA', 'r_pFus', 'l_pFus']
SUBJECT_NAMES = ['by', 'cm', 'cp', 'cyx', 'qd', 'syy', 'xr', 'yyq', 'zdl', 'zhuq']

RESULT_DATA_DIR = "G:/workingdir/BAA/seven/result/"

AC_AVERAGE_RESULT_DOCX_FILE = 'asrg_average.docx'
AC_RESULT_DOCX_FILE = "AC_peak_point_results.docx"
AC_RESULT_NPY_FILE = "AC_peak_point_results.npy"
AC_OPTIMAL_FILE = "AC_optimal_file.nii.gz"

ASRG_RESULT_DOC_DATA_DIR = RESULT_DATA_DIR + 'doc/asrg/'
ANALYSIS_DIR = RESULT_DATA_DIR + 'analysis/'
ASRG_RESULT_AVERAGE_DATA_DIR = 'G:/workingdir/BAA/seven/result/average/asrg/'
ASRG_RESULT_SESSION_AVERAGE_DATA_DIR = 'G:/workingdir/BAA/seven/result/average/asrg/session/'
AC_SESSION_AVERAGE_RESULT_DOCX_FILE = 'asrg_session_average.docx'

RSRG_RESULT_AVERAGE_DATA_DIR = 'G:/workingdir/BAA/seven/result/average/rsrg/'
RSRG_RESULT_DOC_DATA_DIR = RESULT_DATA_DIR + 'doc/rsrg/'
RSRG_RESULT_FILE = 'rsrg_result_file.nii.gz'
RSRG_RESULT_DOCX_FILE = 'rsrg_result_file.docx'

MANUAL_RESULT_DOC_DATA_DIR = RESULT_DATA_DIR + 'doc/manual/'
MANUAL_FILE = ''
GSS_RESULT_DOC_DATA_DIR = RESULT_DATA_DIR + 'doc/gss/'

RSRG_PROB_ROI_202_SUB_FILE = 'G:/nfs/4Ddata/roi_prob/'
RSRG_PROB_ROI_SUBJECT_FILE = 'G:/workingdir/BAA/seven/session/combined_prob/'
RSRG_PROB_ROI_10_SUB_FILE = 'G:/workingdir/BAA/seven/session/t2.3_prob/labels/'
RSRG_RESULT_SESSION_AVERAGE_DATA_DIR = 'G:/workingdir/BAA/seven/result/average/rsrg/session/'


ALL_PROB_MASK = 'G:/nfs/4Ddata/roi_prob/all_prob_mask.nii.gz'
FOUR_D_DATA_DIR = "G:/workingdir/BAA/seven/session/combined/"

TEMP_IMG_DIR = RESULT_DATA_DIR + 'temp.png'
PEAK_POINTS_DIR = 'G:/workingdir/BAA/seven/peak_points/'
COMBINED = 'G:/workingdir/BAA/seven/session/combined/'
SUB_2006_ROI_PROB = 'G:/nfs/4Ddata/roi_prob/'

REGION_VALS = 'region_vals.npy'
PER_VALS = 'per_vals.npy'

ALL_SESSION_AVEARGE_FILE = "all_session_average.nii.gz"
ALL_SESSIONS_AVERAGE_DIR = 'G:/workingdir/BAA/seven/result/average/asrg/session/'
ALL_SESSIONS_AVERAGE_DOCX_FILE = 'asrg_all_sessions_average.docx'

# METHOD_NAMES = ['AC', 'A+B', 'OFFSET', '|A|+|B|', 'OTSU', 'STD']
# METHOD_COLOR_SHAPE = ['bo-', 'rv-', 'g*-', 'y^-', 'kh-', 'mx-']

METHOD_NAMES = ['AC', 'A+B',  'OTSU', 'STD']
METHOD_COLOR_SHAPE = ['bo-', 'rv-', 'g*-', 'mx-']

METHODS_COMPARE_DOCX_FILE = 'methods_compare.docx'
RSRG_RANDOM_WALKER = 'random_walker/'
RSRG_AGGREGATOR_DIR = 'G:/workingdir/BAA/seven/result/average/rsrg/session/random_walker/'

WATERSHED_RESULT_DATA_DIR = RESULT_DATA_DIR + 'doc/watershed/'
WATERSHED_RESULT_FILE = 'watershed.nii.gz'
WATERSHED_GRADIENT_RESULT_FILE = 'watershed_gradient.nii.gz'

RW_RESULT_DATA_DIR = RESULT_DATA_DIR + 'doc/rw/'
RW_AGGRAGATOR_RESULT_DATA_DIR = RW_RESULT_DATA_DIR + 'aggragator/'
RW_RESULT_FILE = 'rw_result_file.nii.gz'
RW_DOCX_RESULT_FILE = 'rw_result_file.docx'
RW_ATLAS_BASED_RESULT_FILE = 'rw_atlas_based_result_file.nii.gz'
RW_ATLAS_BASED_DOCX_RESULT_FILE = 'rw_atlas_based_result_file.docx'
RW_ATLAS_BASED_AGGRATOR_RESULT_FILE = 'rw_atlas_based_aggrator_result_file.nii.gz'
RW_PROB_RESULT_FILE = 'rw_prob_result_file.nii.gz'
RW_PROB_BACKGROUND_RESULT_FILE = 'rw_prob_background.nii.gz'

NEIGHBOUR_SEEDS_DIR = RSRG_RESULT_DOC_DATA_DIR + '26_neighbour_seeds/'

COLOR = ['b', 'r', 'g', 'm']
METHODS = ['Manual', 'GSS', 'AC', 'ARW']


