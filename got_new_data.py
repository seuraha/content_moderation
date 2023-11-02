import pandas as pd
import content_moderation_constants as C
import os

def got_new_ratings_data(num_files):
    cols = [
        'noteId', 'raterParticipantId', 'createdAtMillis', 
        'helpfulnessLevel', 
        'helpfulOther', 'helpfulInformative', 'helpfulClear', 'helpfulEmpathetic', 'helpfulGoodSources', 
        'helpfulUniqueContext', 'helpfulAddressesClaim', 'helpfulImportantContext', 'helpfulUnbiasedLanguage', 
        'notHelpfulOther', 'notHelpfulIncorrect', 'notHelpfulSourcesMissingOrUnreliable', 'notHelpfulOpinionSpeculationOrBias', 
        'notHelpfulMissingKeyPoints', 'notHelpfulOutdated', 'notHelpfulHardToUnderstand', 'notHelpfulArgumentativeOrBiased', 
        'notHelpfulOffTopic', 'notHelpfulSpamHarassmentOrAbuse', 'notHelpfulIrrelevantSources', 'notHelpfulOpinionSpeculation', 'notHelpfulNoteNotNeeded']
    
    pd.read_csv(
        os.path.join(C.data_path, "ratings-00000.tsv"), 
        sep='\t',
        usecols=cols
        ).to_parquet(os.path.join(C.data_path, "ratings-00000.parquet"))
    if num_files > 1:
        for n in range(1, num_files):
            pd.read_csv(
                    os.path.join(C.data_path, f"ratings-0000{n}.tsv"), 
                    sep='\t',
                    usecols=cols
                    ).to_parquet(os.path.join(C.data_path, f"ratings-0000{n}.parquet"))
        
def got_new_user_enrollment_data(num_files):
    pd.read_csv(
        os.path.join(C.data_path, "userEnrollment-00000.tsv"), 
        sep='\t').to_parquet(os.path.join(C.data_path, "userEnrollment-00000.parquet"))
    if num_files > 1:
        for n in range(1, num_files):
            pd.read_csv(
                    os.path.join(C.data_path, f"userEnrollment-0000{n}.tsv"), 
                    sep='\t').to_parquet(os.path.join(C.data_path, f"userEnrollment-0000{n}.parquet"))
            
def got_new_notes_data(num_files):
    pd.read_csv(
        os.path.join(C.data_path, "notes-00000.tsv"),
        sep='\t').to_parquet(os.path.join(C.data_path, "notes-00000.parquet"))
    if num_files > 1:
        for n in range(1, num_files):
            pd.read_csv(
                    os.path.join(C.data_path, f"notes-0000{n}.tsv"),
                    sep='\t').to_parquet(os.path.join(C.data_path, f"notes-0000{n}.parquet"))
    