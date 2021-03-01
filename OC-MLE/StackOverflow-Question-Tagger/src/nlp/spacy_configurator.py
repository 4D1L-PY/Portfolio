from requests import get
from src.nlp.text_preprocessor import BeautifulSoup


class SpacyConfigurator:

    def __init__(self, language, tags_url='https://spacy.io/api/annotation'):
        """

        :param language:
        :param tags_url:
        """
        self.language = language
        self.tags_url = tags_url
        self.pipeline_tags = ['tagger',
                              'parser',
                              'ner',
                              'textcat',
                              'entity_linker',
                              'entity_ruler',
                              'sentencizer',
                              'merge_noun_chunks',
                              'merge_entities',
                              'merge_subtokens']
        self.tags = self.get_tags()
        # To do : add more tag types

    def get_tags(self):
        """

        :return:
        """
        html_soup = BeautifulSoup(get(self.tags_url).text, 'html.parser')
        # Tagging type titles
        title_parser = html_soup.find_all('a', {"class": "heading-text e80ba60d"})[4:-3]
        tagging_type_titles = [title.text.strip() for title in title_parser]
        # Tagging type subtitles
        subtitle_patterns = ['Universal', 'English', 'German']
        subtitle_parser = html_soup.find_all('span', {"class": "heading-text"})
        subtitles = [t.text for t in subtitle_parser for s in subtitle_patterns if s in t.text]
        # Tagging tables
        tagging_tables_parser = html_soup.find_all('table', {"class": "_59fbd182"})
        tagging_tables = [pos_table for pos_table in tagging_tables_parser]
        # Select relevant titles
        rebuilt_titles = subtitles + tagging_type_titles[2:]
        # Tags html tag class pattern
        tags_cls_pattern = {"class": "_1d7c6046"}
        spacy_tags = {}
        for i, title in enumerate(rebuilt_titles):
            tags = [t.text for t in tagging_tables[i].find_all('code', tags_cls_pattern) if '=' not in t.text]
            tags = list(set(tags))
            if i in [1, 2]:
                spacy_tags[f'{title} Part-of-speech Tags'] = tags
            elif i in [4, 5]:
                spacy_tags[f'{title} Dependency Labels'] = tags
            else:
                spacy_tags[title] = tags
        return spacy_tags
