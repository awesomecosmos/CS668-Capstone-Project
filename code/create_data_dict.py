import pickle
import xmltodict

# reading xml data dictionary
with open('data/highered_00001.xml', 'r', encoding='utf-8') as file:
    xml = file.read()

# extracting data from xml data dictionary
parsed_xml = xmltodict.parse(xml) 

# re-coded and simplified data dictionary
full_data_dict = {}

# extracting element in parsed_xml which contains the data we need
data_dict_to_extract = parsed_xml['codeBook']['dataDscr']['var']

# iterating over all variables (features)
for i in range(len(data_dict_to_extract)):
    # settting feature name as dictionary key
    var_name_key = data_dict_to_extract[i]['@ID']

    # creating dictionary to store data for current feature
    var_data_dict = {}
    # setting values in dictionary for feature
    var_data_dict['var_desc_short'] = data_dict_to_extract[i]['labl']
    var_data_dict['var_desc'] = data_dict_to_extract[i]['txt']

    # creating dictionary to store information about data values
    data_values = {}
    try:
        for dict in data_dict_to_extract[i]['catgry']:
            key = dict['catValu']
            value = dict['labl']
            try:
                # if data value is numeric, store it as numeric
                data_values[int(key)] = value
            except:
                # otherwise store as-is
                data_values[key] = value
    except:
        # if no data value information provided, leave blank
        pass

    # assigning dictionary with data values to its key
    var_data_dict['data_values'] = data_values

    # assigning dictionary of all data/metadata for current feature
    full_data_dict[var_name_key] = var_data_dict

# dumping final data dictionary to pickle
with open('data/data_dictionary.pkl', 'wb') as handle:
    pickle.dump(full_data_dict, handle)