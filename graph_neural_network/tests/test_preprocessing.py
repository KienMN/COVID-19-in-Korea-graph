import pytest
from graph_neural_network.preprocessing import *

def test_load_data():
  patient_info_df, case_df = load_patient_and_case_data()
  assert patient_info_df.shape == (5164, 14)
  assert case_df.shape == (174, 8)

patient_info_df, case_df = load_patient_and_case_data()

def test_get_patient_patient_matrix():
  p_infected_by_p = get_patent_patient_matrix(patient_info_df)
  assert p_infected_by_p.shape == (5164, 5164)
  assert p_infected_by_p.count_nonzero() == 1340

def test_get_patient_province_matrix():
  p_lives_in_p = get_patient_province_matrix(patient_info_df)
  assert p_lives_in_p.shape == (5164, 17)
  assert p_lives_in_p.count_nonzero() == 5164

def test_get_infection_case_province_matrix():
  infection_case_in_province = get_infection_case_province_matrix(case_df, patient_info_df)
  assert infection_case_in_province.shape == (81, 17)
  assert infection_case_in_province.count_nonzero() == 174

def test_patient_infection_case_matrix():
  patient_related_to_case = get_patient_infection_case_matrix(case_df, patient_info_df)
  assert patient_related_to_case.shape == (5164, 81)
  assert patient_related_to_case.count_nonzero() == 4245

def test_get_city_province_matrix():
  city_located_in_province, patient_lives_in_city, case_in_city = get_city_province_matrix(case_df, patient_info_df)
  assert city_located_in_province.shape == (187, 17)
  assert city_located_in_province.count_nonzero() == 187
  assert patient_lives_in_city.shape == (5164, 187)
  assert patient_lives_in_city.count_nonzero() == 5070
  assert case_in_city.shape == (81, 187)
  assert case_in_city.count_nonzero() == 67

def test_get_training_data():
  patient_related_to_case = get_patient_infection_case_matrix(case_df, patient_info_df)
  patient_idx, labels = get_training_data(patient_related_to_case)
  assert len(labels) == 5164
  assert (labels != -1).sum() == 3828
  assert labels.min() == -1
  assert labels.max() == 9
  assert (patient_idx == patient_related_to_case.tocsr()[:, [76, 80, 77, 34, 50, 26, 62, 6, 71, 10]].tocoo().row).all()

def test_post_processing():
  assert patient_info_df.shape == (5164, 14)
  assert case_df.shape == (174, 8)