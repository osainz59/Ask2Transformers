from typing import List

from . import NLITopicClassifierWithMappingHead

import numpy as np

BABELDOMAINS_TOPICS = [
	"Animals",
	"Art, architecture, and archaeology",
	"Biology",
	"Business, economics, and finance",
	"Chemistry and mineralogy",
	"Computing",
	"Culture and society",
	"Education",
	"Engineering and technology",
	"Farming",
	"Food and drink",
	"Games and video games",
	"Geography and places",
	"Geology and geophysics",
	"Health and medicine",
	"Heraldry, honors, and vexillology",
	"History",
	"Language and linguistics",
	"Law and crime",
	"Literature and theatre",
	"Mathematics",
	"Media",
	"Meteorology",
	"Music",
	"Numismatics and currencies",
	"Philosophy and psychology",
	"Physics and astronomy",
	"Politics and government",
	"Religion, mysticism and mythology",
	"Royalty and nobility",
	"Sport and recreation",
	"Textile and clothing",
	"Transport and travel",
	"Warfare and defense"
]

BABELDOMAINS_TOPIC_MAPPING = {
	"Animals": "Animals",
	"Art": "Art, architecture, and archaeology",
	"Architecture": "Art, architecture, and archaeology",
	"Archaeology": "Art, architecture, and archaeology",
	"Art, architecture, and archaeology": "Art, architecture, and archaeology",
	"Biology": "Biology",
	"Business": "Business, economics, and finance",
	"Economics": "Business, economics, and finance",
	"Finance": "Business, economics, and finance",
	"Business, economics, and finance": "Business, economics, and finance",
	"Chemistry": "Chemistry and mineralogy",
	"Mineralogy": "Chemistry and mineralogy",
	"Chemistry and mineralogy": "Chemistry and mineralogy",
	"Computing": "Computing",
	"Culture": "Culture and society",
	"Society": "Culture and society",
	"Culture and society": "Culture and society",
	"Education": "Education",
	"Engineering": "Engineering and technology",
	"Technology": "Engineering and technology",
	"Engineering and technology": "Engineering and technology",
	"Farming": "Farming",
	"Food": "Food and drink",
	"Drink": "Food and drink",
	"Food and drink": "Food and drink",
	"Games": "Games and video games",
	"Video games": "Games and video games",
	"Games and video games": "Games and video games",
	"Geography": "Geography and places",
	"Places": "Geography and places",
	"Geography and places": "Geography and places",
	"Geology": "Geology and geophysics",
	"Geophysics": "Geology and geophysics",
	"Geology and geophysics": "Geology and geophysics",
	"Health": "Health and medicine",
	"Medicine": "Health and medicine",
	"Health and medicine": "Health and medicine",
	"Heraldry": "Heraldry, honors, and vexillology",
	"Honors": "Heraldry, honors, and vexillology",
	"Vexillology": "Heraldry, honors, and vexillology",
	"Heraldry, honors, and vexillology": "Heraldry, honors, and vexillology",
	"History": "History",
	"Language": "Language and linguistics",
	"Linguistics": "Language and linguistics",
	"Language and linguistics": "Language and linguistics",
	"Law": "Law and crime",
	"Crime": "Law and crime",
	"Law and crime": "Law and crime",
	"Literature": "Literature and theatre",
	"Theatre": "Literature and theatre",
	"Literature and theatre": "Literature and theatre",
	"Mathematics": "Mathematics",
	"Media": "Media",
	"Meteorology": "Meteorology",
	"Music": "Music",
	"Numismatics": "Numismatics and currencies",
	"Currencies": "Numismatics and currencies",
	"Numismatics and currencies": "Numismatics and currencies",
	"Philosophy": "Philosophy and psychology",
	"Psychology": "Philosophy and psychology",
	"Philosophy and psychology": "Philosophy and psychology",
	"Physics": "Physics and astronomy",
	"Astronomy": "Physics and astronomy",
	"Physics and astronomy": "Physics and astronomy",
	"Politics": "Politics and government",
	"Goverment": "Politics and government",
	"Politics and government": "Politics and government",
	"Religion": "Religion, mysticism and mythology",
	"Mysticism": "Religion, mysticism and mythology",
	"Mythology": "Religion, mysticism and mythology",
	"Religion, mysticism and mythology": "Religion, mysticism and mythology",
	"Royalty": "Royalty and nobility",
	"Nobility": "Royalty and nobility",
	"Royalty and nobility": "Royalty and nobility",
	"Sport": "Sport and recreation",
	"Recreation": "Sport and recreation",
	"Sport and recreation": "Sport and recreation",
	"Textile": "Textile and clothing",
	"Clothing": "Textile and clothing",
	"Textile and clothing": "Textile and clothing",
	"Transport": "Transport and travel",
	"Travel": "Transport and travel",
	"Transport and travel": "Transport and travel",
	"Warfare": "Warfare and defense",
	"Defense": "Warfare and defense",
	"Warfare and defense": "Warfare and defense"
}


class BabelDomainsClassifier(NLITopicClassifierWithMappingHead):
	""" BabelDomainsClassifier

	Specific class for topic classification using BabelDomains topic set.
	"""

	def __init__(self, **kwargs):
		super(BabelDomainsClassifier, self).__init__(
			pretrained_model='roberta-large-mnli', topics=BABELDOMAINS_TOPICS, topic_mapping=BABELDOMAINS_TOPIC_MAPPING,
			query_phrase="The domain of the sentence is about", entailment_position=2, **kwargs)

		def idx2topic(idx):
			return BABELDOMAINS_TOPICS[idx]

		self.idx2topic = np.vectorize(idx2topic)

	def predict_topics(self, contexts: List[str], batch_size: int = 1, return_labels: bool = True,
	                   return_confidences: bool = False, topk: int = 1):
		output = self(contexts, batch_size)
		topics = np.argsort(output, -1)[:, ::-1][:, :topk]
		if return_labels:
			topics = self.idx2topic(topics)
		if return_confidences:
			topics = np.stack((topics, np.sort(output, -1)[:, ::-1][:, :topk]), -1).tolist()
			#topics = [list(map(tuple, row)) for row in topics]
			topics = [[(int(label), conf) if not return_labels else (label, conf) for label, conf in row ]
			          for row in topics]
		else:
			topics = topics.tolist()
		if topk == 1:
			topics = [row[0] for row in topics]

		return topics
