import sys
import pandas as pd

if __name__ == "__main__":
    folderpath = sys.argv[1]
    dataset_paths = datapaths(folderpath)
    ## Unique identifier to id the ones that we care about:
    ## Annoying: First get the number and positions of all datasets:

    numbers = [int(d.split('cropped_part')[-1].split('DeepCut')[0]) for d in dataset_paths if unique_string in d]
    max_ind = np.max(numbers)

    ## Behavior excel spreadsheet name: 
    sheet_tag = sys.argv[2]
    ## Write out some important strings: 
    behavior_tag = 'Behavior'
    start_tag = "Start (s)"
    stop_tag = "Stop (s)"

    dam_pos_tag = "Mother in nest"
    virg_pos_tag = "Virgin in rest"
    pursuit_tag = "Mom agressing"

    ## Get the spreadsheet: 
    excel_data = pd.read_excel(sheet_tag)

    ## Just get out the parts we care about: 
    trimmed_data = excel_data[[behavior_tag,start_tag,stop_tag]]

    ## Further separate out into the behaviors we care about 
    dam_pos = trimmed_data[trimmed_data[behavior_tag] == dam_pos_tag][[start_tag,stop_tag]]
    virg_pos = trimmed_data[trimmed_data[behavior_tag] == virg_pos_tag][[start_tag,stop_tag]]
    pursuit = trimmed_data[trimmed_data[behavior_tag] == pursuit_tag][[start_tag,stop_tag]]
    
    ## Now package up the starts and stops : 
    ethogram_sources = [dam_pos,virg_pos,pursuit]
    ethogram_name = ['Mother,Virgin,Pursuit']
    for s,source in enumerate(ethogram_sources):
        ## initialize ethogram: 
        ethogram  = np.zeros((1+max_ind)*36000,)*np.nan
        for ind,ent in dam_pos.iterrows():
            start,end = 30*ent[start_tag],30*ent[stop_tag]
            ethogram[start:end] = 1
            
            print(ent[start_tag],ent[stop_tag])
        namestring = folderpath+'/'+'dataset_'+ethogram_name[s]+'gt_ethogram'
        joblib.dump(ethogram,namestring)

    
