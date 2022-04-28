import unittest

from a2t.data import (
    TACREDRelationClassificationDataset,
    WikiEventsArgumentClassificationDataset,
    ACEArgumentClassificationDataset,
)
from a2t.tasks.tuple_classification import (
    TACREDRelationClassificationTask,
    TACREDFeatures,
    EventArgumentClassificationTask,
    EventArgumentClassificationFeatures,
)

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


class TestWikiEventsDatasets(unittest.TestCase):
    """TODO: Implement test cases"""

    @unittest.skipIf(not os.path.exists("data/wikievents/dev.jsonl"), "No WikiEvents data available at data/wikievents/")
    def test_data_loader(self):

        WikiEvents_LABELS = ["Victim", "no_relation"]

        # Test class creation without errors
        dataset = WikiEventsArgumentClassificationDataset("data/wikievents/dev.jsonl", labels=WikiEvents_LABELS)

        # Test instance class is the correct
        self.assertTrue(isinstance(dataset[0], EventArgumentClassificationFeatures))

        # Test that instances are loaded correctly
        self.assertTrue(dataset[0].label == "Victim")

        # Test that above is true for all the instances
        task = EventArgumentClassificationTask(
            "WikiEvents EAE", labels=WikiEvents_LABELS, templates={"Victim": ["{arg} is a victim."]}
        )

        task.assert_features_class(dataset)


class TestACEDatasets(unittest.TestCase):
    """TODO: implement test cases"""

    @unittest.skipIf(not os.path.exists("data/ace/dev.oneie.json"), "No WikiEvents data available at data/ace/")
    def test_data_loader(self):

        ACE_LABELS = ["Artifact", "no_relation"]

        # Test class creation without errors
        dataset = ACEArgumentClassificationDataset("data/ace/dev.oneie.json", labels=ACE_LABELS)

        # Test instance class is the correct
        self.assertTrue(isinstance(dataset[0], EventArgumentClassificationFeatures))

        # Test that instances are loaded correctly
        self.assertTrue(dataset[0].label == "Artifact")

        # Test that above is true for all the instances
        task = EventArgumentClassificationTask(
            "WikiEvents EAE", labels=ACE_LABELS, templates={"Artifact": ["{arg} is an artifact."]}
        )

        task.assert_features_class(dataset)
