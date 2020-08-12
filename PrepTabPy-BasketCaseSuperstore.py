import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import mlxtend as ml


def createtable(DataFrame):
    frequent_itemsets = apriori(DataFrame, min_support=0.10, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric='lift')
    #rules.sort_values('confidence', ascending = False, inplace = True)
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
    rules["consequents"] = rules["consequents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
    return rules[["antecedents","consequents","lift","confidence","support"]]



def get_output_schema():       
  return pd.DataFrame({
    'antecedents' : prep_string(),
    'consequents' : prep_string(),
    'lift' : prep_decimal(),
    'confidence' : prep_decimal(),
    'support' : prep_decimal(),
})