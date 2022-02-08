import unittest

from a2t.data.tacred import TACREDRelationClassificationDataset
from a2t.tasks.tuple_classification import TACREDRelationClassificationTask, TACREDFeatures

import os


class TestTACREDRelationClassificationDataset(unittest.TestCase):
    @unittest.skipIf(not os.path.exists("data/tacred/dev.json"), "No TACRED data available at data/tacred/")
    def test_data_loader(self):

        TACRED_LABELS = [
            "no_relation",
            "org:alternate_names",
            "org:city_of_headquarters",
            "org:country_of_headquarters",
            "org:dissolved",
            "org:founded",
            "org:founded_by",
            "org:member_of",
            "org:members",
            "org:number_of_employees/members",
            "org:parents",
            "org:political/religious_affiliation",
            "org:shareholders",
            "org:stateorprovince_of_headquarters",
            "org:subsidiaries",
            "org:top_members/employees",
            "org:website",
            "per:age",
            "per:alternate_names",
            "per:cause_of_death",
            "per:charges",
            "per:children",
            "per:cities_of_residence",
            "per:city_of_birth",
            "per:city_of_death",
            "per:countries_of_residence",
            "per:country_of_birth",
            "per:country_of_death",
            "per:date_of_birth",
            "per:date_of_death",
            "per:employee_of",
            "per:origin",
            "per:other_family",
            "per:parents",
            "per:religion",
            "per:schools_attended",
            "per:siblings",
            "per:spouse",
            "per:stateorprovince_of_birth",
            "per:stateorprovince_of_death",
            "per:stateorprovinces_of_residence",
            "per:title",
        ]

        # Test class creation without errors
        dataset = TACREDRelationClassificationDataset("data/tacred/dev.json", labels=TACRED_LABELS)

        # Test instance class is the correct
        self.assertTrue(isinstance(dataset[0], TACREDFeatures))

        # Test that instances are loaded correctly
        self.assertTrue(dataset[0].label == "per:title")

        # Test that above is true for all the instances
        task = TACREDRelationClassificationTask(TACRED_LABELS, {"per:title": ["{subj} is also known as {obj}"]}, None)

        task.assert_features_class(dataset)
