import numpy as np
from cc3d import connected_components

def test_overlap(pointlist1,pointlist2):
    ''' 
    test_result = test_overlap(pointlist1,pointlist2)
    
    Checks whether any point in pointlist1 is also in pointlist2
    '''
    # First check trivial cases with fast methods
    pa1 = np.asarray(pointlist1)
    pa2 = np.asarray(pointlist2)
    ndims = pa1.shape[1]
    if(np.min(pa1[:,0]) > np.max(pa2[:,0])): return False
    if(np.min(pa2[:,0]) > np.max(pa1[:,0])): return False
    if(np.min(pa1[:,1]) > np.max(pa2[:,1]) and ndims >= 2): return False
    if(np.min(pa2[:,1]) > np.max(pa1[:,1]) and ndims >= 2): return False
    if(np.min(pa1[:,2]) > np.max(pa2[:,2]) and ndims >= 3): return False
    if(np.min(pa2[:,2]) > np.max(pa1[:,2]) and ndims >= 3): return False
    # Then check point by point overlap
    test_result = False
    maxind = len(pointlist1) - 1
    ind = 0
    while(test_result == False and ind <= maxind):
        if(pointlist1[ind] in pointlist2): test_result = True
        ind += 1
    return test_result

def get_blobs(volume):
    '''
    bloblist = get_blobs(volume)
    
    This function returns a list of dictionaries, in which each dictionary
    represents one blob in the given 'searchvolume'. A blob is defined as 
    a set of connected points. The 'searchvolume' is expected to be a 
    p-dimensional Numpy array of zero and non-zero values. All neighboring
    non-zero values will be treated as connected points, i.e. a blob.
    
    Each blob dictionary in the list 'blobs' has the following entries:
        * blob['id'] - Number of blob in searchvolume, starting with 0
        * blob['points'] - List of points in this blob. Each point is a 1D Numpy array with p coordinates (one per dimension)
        
    NB: The runtime of this function is largely independent of size of the 
    searchvolume, but grows with the number as well as the size of blobs.
    For busy 3D volumes, get_blobs_fast() can >100 times faster (but might
    falsly merge two almost-overlapping points in rare cases)
    
    This version is using an external library for connected components (26-connectedness)
    that was not available at the beginning of Project Leo. Please see:
        https://github.com/seung-lab/connected-components-3d
    '''
    # print("Performing cca...")
    #todo wieder raus
    if True:#np.amax(volume) == 1:
        volume = volume.astype(np.bool_)
        labeled_volume = connected_components(volume)
    else:
        print("\tUsing predefined labels...")
        labeled_volume = volume
    labels = [ x for x in np.unique(labeled_volume) if x != 0 ]
    bloblist = []
    for label in labels:
        allpoints = np.asarray(np.where(labeled_volume == label)).T.tolist() # returns list of pointers; slow for large vols
        blob = {}
        blob['id'] = len(bloblist)
        blob['points'] = allpoints
        bloblist.append(blob)
    return bloblist

def get_patch_overlap(pred_patch, target_patch):
    """Test for overlap in two patches
    Returns:
        true positive, true negative, false positive, false negative
    """
    pred = get_blobs(pred_patch)
    target = get_blobs(target_patch)
    tp, fp, fn = 0, 0, 0
    # result_dict = {}
    hit_target = set()
    for blob_pred in pred:
        hits_i = 0
        pred_l = blob_pred["points"]
        pred_id = blob_pred["id"]
        for blob_target in target:
            target_l = blob_target["points"]
            target_id = blob_target["id"]
            # print(f"Testing {pred_id} {target_id}",end="\r",flush=True)
            if test_overlap(pred_l, target_l):
                hits_i += 1
                hit_target.add(target_id)
        if hits_i == 0 or hits_i > 1:
            fp += 1
        else:
            tp += hits_i
    fn = len(target) - len(hit_target)
    return tp, fp, fn
