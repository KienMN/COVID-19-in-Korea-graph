import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
import time
import dgl
import os

def load_patient_and_case_data():
  """Load patient information and infection case (cluter) data frame.
  
  Returns
  -------
  patient_info_df: pd.DataFrame shape of (n_patients, n_features_of_patient)
    The information about patients.

  case_df: pd.DataFrame shape of (n_cases, n_features_of_case)
    The information about infection cases (cluster).
  """
  filedir = os.path.dirname(__file__) + '/data/'
  patient_info_df = pd.read_csv(filedir + 'PatientInfo.csv')
  case_df = pd.read_csv(filedir + 'Case.csv')

  # Delete duplicated `patient_id`
  patient_info_df = patient_info_df[~patient_info_df['patient_id'].duplicated(keep='first')]
  # Sorted by `patient_id`
  patient_info_df = patient_info_df.sort_values(by='patient_id')
  patient_info_df = patient_info_df.astype({'patient_id': 'category',
                                            'province': 'category',
                                            'infection_case': 'category'})
  patient_info_df = patient_info_df.reset_index(drop=True)

  # Synchronize code between patient_info_df and case_df
  case_df = case_df.astype({'province': 'category',
                            'infection_case': 'category'})
  case_df['province'].cat.set_categories(patient_info_df['province'].cat.categories,
                                         inplace=True)
  patient_info_df['infection_case'].cat.set_categories(case_df['infection_case'].cat.categories,
                                                       inplace=True)

  # Delete unknown value between patient_info_df and case_df
  # patient_info_df = patient_info_df[patient_info_df['infection_case'].cat.codes != -1]
  case_df = case_df[case_df['province'].cat.codes != -1]

  return patient_info_df, case_df

def get_patent_patient_matrix(patient_info_df):
  """Create patient - infected by - patient matrix.
  
  Parameters
  ----------
  patient_info_df: pd.DataFrame shape of (n_patients, n_features_of_patient)
    The information about patients.

  Returns
  -------
  matrix: scipy sparse COO matrix of shape (n_patients, n_patients)
    Matrix of relationship between pair of patients
    1 indicates that a patient infected the other patient.
  """
  related_features = ['patient_id', 'infected_by']
  patient_info_df = patient_info_df[related_features]
  infection_rel = patient_info_df[~patient_info_df['infected_by'].isna()]
  uids = []
  vids = []
  for i in range(infection_rel.shape[0]):
    pid = infection_rel.iloc[i, 0]
    # Get code of patient instead of patient_id.
    u = infection_rel[infection_rel['patient_id'] == pid]['patient_id'].cat.codes.values[0]
    infected_by_pids = [int(i) for i in infection_rel.iloc[i, 1].split(',')]
    for ibpid in infected_by_pids:
      if len(patient_info_df[patient_info_df['patient_id'] == ibpid]) != 0:
        # Check existance of user in the dataset.
        # Get code of patient.
        v = patient_info_df[patient_info_df['patient_id'] == ibpid]['patient_id'].cat.codes.values[0]
        uids.append(u)
        vids.append(v)
  
  # Test the number of relationship.
  assert len(uids) == len(vids) == 1340

  # Adjacency matrix for patient - infected by - patient relationship.
  p_infected_by_p = sp.coo_matrix((np.ones(len(uids)), (uids, vids)),
                                  shape=(patient_info_df.shape[0], patient_info_df.shape[0]))
  return p_infected_by_p

def get_patient_province_matrix(patient_info_df):
  """Create patient - province matrix.
  
  Parameters
  ----------
  patient_info_df: pd.DataFrame shape of (n_patients, n_features_of_patient)
    The information about patients.

  Returns
  -------
  matrix: scipy sparse COO matrix of shape (n_patients, n_provinces)
    Matrix of relationship between pair of patient and province.
    1 indicates that a patient lives in a province.
  """

  related_features = ['patient_id', 'province']
  patient_info_df = patient_info_df[related_features]
  n_patients = patient_info_df.shape[0]
  n_provinces = len(patient_info_df['province'].cat.categories)
  uids = []
  vids = []

  for i in range(n_patients):
    pid = patient_info_df.iloc[i, 0]
    u = patient_info_df[patient_info_df['patient_id'] == pid]['patient_id'].cat.codes.values[0]
    v = patient_info_df[patient_info_df['patient_id'] == pid]['province'].cat.codes.values[0]
    uids.append(u)
    vids.append(v)

  p_lives_in_p = sp.coo_matrix((np.ones(len(uids)), (uids, vids)),
                               shape=(n_patients, n_provinces))
  return p_lives_in_p

def get_infection_case_province_matrix(case_df, patient_info_df):
  """Create infection case - province matrix

  Parameters
  ----------
  case_df: pd.DataFrame shape of (n_cases, n_features_of_case)
    The information about infection cases (cluster).

  patient_info_df: pd.DataFrame shape of (n_patients, n_features_of_patient)
    The information about patients.
  
  Returns
  -------
  matrix: scipy sparse COO matrix of shape (n_infection_cases, n_provinces)
    Matrix of relationship between pair of case (cluster) and province.
    1 indicates that a cluster occurs in a province.
  """

  n_infection_cases = len(case_df['infection_case'].cat.categories)
  n_provinces = len(patient_info_df['province'].cat.categories)
  uids = []
  vids = []

  for i in range(case_df.shape[0]):
    row = case_df.iloc[[i], :]
    u = row['infection_case'].cat.codes.values[0]
    v = row['province'].cat.codes.values[0]
    if u != -1 and v != -1:
      uids.append(u)
      vids.append(v)

  infection_case_in_province = sp.coo_matrix((np.ones(len(uids)), (uids, vids)),
                                             shape=(n_infection_cases, n_provinces))
  return infection_case_in_province

def get_patient_infection_case_matrix(case_df, patient_info_df):
  """Create patient - infection case matrix

  Parameters
  ----------
  case_df: pd.DataFrame shape of (n_cases, n_features_of_case)
    The information about infection cases (cluster).
  
  patient_info_df: pd.DataFrame shape of (n_patients, n_features_of_patient)
    The information about patients.
  
  Returns
  -------
  matrix: scipy sparse COO matrix of shape (n_patients, n_infection_cases)
    Matrix of relationship between pair of patient and infection case.
    1 indicates that a patient relates to a cluster.
  """
  n_patients = patient_info_df.shape[0]
  n_infection_cases = len(case_df['infection_case'].cat.categories)
  uids = []
  vids = []

  for i in range(n_patients):
    row = patient_info_df.iloc[[i], :]
    u = row['patient_id'].cat.codes.values[0]
    v = row['infection_case'].cat.codes.values[0]
    if v != -1:
      uids.append(u)
      vids.append(v)

  patient_related_to_case = sp.coo_matrix((np.ones(len(uids)), (uids, vids)),
                                          shape=(n_patients, n_infection_cases))
  return patient_related_to_case

def get_city_province_matrix(case_df, patient_info_df):
  """Create patient - infection case matrix

  Parameters
  ----------
  case_df: pd.DataFrame shape of (n_cases, n_features_of_case)
    The information about infection cases (cluster).
  
  patient_info_df: pd.DataFrame shape of (n_patients, n_features_of_patient)
    The information about patients.
  
  Returns
  -------
  city_located_in_province: scipy sparse COO matrix of shape (n_cities, n_provinces)
    Matrix of relationship between pair of city and province.
    1 indicates that a city locates in a province.
  
  patient_lives_in_city: scipy sparse COO matrix of shape (n_patients, n_cities)
    Matrix of relationship between pair of patient and city.
    1 indicates that a patient lives in a city.
  
  case_in_city: scipy sparse COO matrix of shape (n_infection_cases, n_cities)
    Matrix of relationship between pair of infection case and city.
    1 indicates that a infection case (cluster) occurs in a city.
  """

  tmp_city = patient_info_df[~patient_info_df['city'].isna()][['patient_id', 'province', 'city']].reset_index(drop=True)
  tmp_city['province_city'] = tmp_city[['province', 'city']].apply(lambda x: x[0] + ', ' + x[1],
                                                                   axis=1)
  tmp_city = tmp_city.astype({'province_city': 'category'})

  tmp_case_city = case_df[~case_df['city'].isna()][['infection_case', 'province', 'city']]
  tmp_case_city['province_city'] = tmp_case_city[['province', 'city']].apply(lambda x: x[0] + ', ' + x[1],
                                                                             axis=1)
  tmp_case_city = tmp_case_city.astype({'province_city': 'category'})
  tmp_case_city['province_city'].cat.set_categories(tmp_city['province_city'].cat.categories,
                                                    inplace=True)

  n_cities = len(tmp_city['province_city'].cat.categories)
  n_provinces = len(patient_info_df['province'].cat.categories)
  n_patients = patient_info_df.shape[0]
  n_infection_cases = len(case_df['infection_case'].cat.categories)
  
  # City - Province
  uids = []
  vids = []
  cities = tmp_city['province_city'].cat.categories

  for city in cities:
    rows = tmp_city[tmp_city['province_city'] == city]
    u = rows['province_city'].cat.codes.values[0]
    v = rows['province'].cat.codes.values[0]
    
    if v != -1:
      uids.append(u)
      vids.append(v)

  city_located_in_province = sp.coo_matrix((np.ones(len(uids)), (uids, vids)),
                                           shape=(n_cities, n_provinces))

  # Patient - City
  uids = []
  vids = []

  for i in range(tmp_city.shape[0]):
    row = tmp_city.iloc[[i], :]
    u = row['patient_id'].cat.codes.values[0]
    v = row['province_city'].cat.codes.values[0]
    if v != -1:
      uids.append(u)
      vids.append(v)

  patient_lives_in_city = sp.coo_matrix((np.ones(len(uids)), (uids, vids)),
                                        shape=(n_patients, n_cities))

  # Infection case - City
  uids = []
  vids = []

  for i in range(tmp_case_city.shape[0]):
    row = tmp_case_city.iloc[[i], :]
    u = row['infection_case'].cat.codes.values[0]
    v = row['province_city'].cat.codes.values[0]
    if v != -1:
      uids.append(u)
      vids.append(v)

  case_in_city = sp.coo_matrix((np.ones(len(uids)), (uids, vids)),
                               shape=(n_infection_cases, n_cities))
  
  return city_located_in_province, patient_lives_in_city, case_in_city

def get_training_data(patient_related_to_case,
                      infection_cases_select=[76, 80, 77, 34, 50, 26, 62, 6, 71, 10],
                      include_rest=False,
                      include_unlabel=False):
  """
  Getting patient index and label of infection case.

  Parameters
  ----------
  patient_related_to_case : scipy sparse COO matrix of shape (n_patients, n_infection_cases)
    Matrix indicating the 1-1 relationship between patient and infection case.

  infection_cases_select : array, default=[76, 80, 77, 34, 50, 26, 62, 6, 71, 10]
    Sequence number of infection case to be selected. Default is top-10 popular cases.

  include_rest : boolean, default: False
    Including all infrequent cases as a new class.

  include_unlabel : boolean, default: False
    Including unknown case as a new class.

  Returns
  -------
  patient_idx : 1D numpy array of shape (n_selected_patients,)
    Indices of selected patients.
  
  labels: 1D numpy array of shape (n_patients,)
    Labels of selected patients. Labels of unselected patients are marked as -1.
  """
  
  n_patients, n_infection_cases = patient_related_to_case.shape
  
  # Labels of patients' infection case.
  labels = np.array([-2] * n_patients)
  labels[patient_related_to_case.row] = patient_related_to_case.col

  # Map label of infrequent classes to -1
  for label in range(n_infection_cases):
    if label not in infection_cases_select:
      labels[labels == label] = -1
  
  # Include all infrequent classes as a new class
  if include_rest:
    labels += 1
  # Include unknown class as a new class
  if include_unlabel:
    labels += 1
  labels[labels < 0] = -1

  # Map labels to 0, 1, 2, ...
  map_labels = dict(zip(np.unique(labels), np.arange(len(np.unique(labels))) - 1))
  labels = np.vectorize(map_labels.get)(labels)
  patient_idx = np.where(labels >= 0)[0]
  return patient_idx, labels

def get_heterogeneous_graph(include_rest=False, include_unlabel=False):
  """
  Get hetorogenous graph for training a GNN.

  Parameters
  ----------
  include_rest: boolean, default: False
    Including all infrequent cases as a new class.

  include_unlabel: boolean, default: False
    Including unknown case as a new class.

  Returns
  -------
  G: DGL Heterograph
    Graph data structure, including different types of nodes and relationships.

  patient_idx : 1D numpy array of shape (n_selected_patients,)
    Indices of selected patients.
  
  labels: 1D numpy array of shape (n_patients,)
    Labels of selected patients. Labels of unselected patients are marked as -1.
  """
  patient_info_df, case_df = load_patient_and_case_data()
  p_infected_by_p = get_patent_patient_matrix(patient_info_df)
  p_lives_in_p = get_patient_province_matrix(patient_info_df)
  infection_case_in_province = get_infection_case_province_matrix(case_df,
                                                                  patient_info_df)
  patient_related_to_case = get_patient_infection_case_matrix(case_df,
                                                              patient_info_df)
  city_located_in_province, patient_lives_in_city, case_in_city = get_city_province_matrix(case_df, patient_info_df)
  patient_idx, labels = get_training_data(patient_related_to_case,
                                          include_rest=include_rest,
                                          include_unlabel=include_unlabel)

  G = dgl.heterograph({
    ('patient', 'infected_by', 'patient'): p_infected_by_p,
    ('patient', 'infected_to', 'patient'): p_infected_by_p.transpose(),
    ('patient', 'lives_in', 'province'): p_lives_in_p,
    ('province', 'manages', 'patient'): p_lives_in_p.transpose(),
    ('infection_case', 'in', 'province'): infection_case_in_province,
    ('province', 'has', 'infection_case'): infection_case_in_province.transpose(),
    ('patient', 'lives_in_city', 'city'): patient_lives_in_city,
    ('city', 'city_has_patient', 'patient'): patient_lives_in_city.transpose(),
    ('city', 'locates', 'province'): city_located_in_province,
    ('province', 'has_city', 'city'): city_located_in_province.transpose(),
    ('infection_case', 'in_city', 'city'): case_in_city,
    ('city', 'has_case', 'infection_case'): case_in_city.transpose(),
  })

  return G, patient_idx, labels