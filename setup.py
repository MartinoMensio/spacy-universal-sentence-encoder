from setuptools import setup, find_packages
from setuptools.command.install import install
from subprocess import getoutput

# Allow installing dependencies not hosted on pyPI
# https://github.com/BaderLab/saber/issues/35#issuecomment-467827175
class PostInstall(install):
    pkgs = ['https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz#en_core_web_sm-2.2.0',
           ' https://github.com/explosion/spacy-models/releases/download/xx_ent_wiki_sm-2.2.0/xx_ent_wiki_sm-2.2.0.tar.gz#xx_ent_wiki_sm-2.2.0']
    def run(self):
        install.run(self)
        print(getoutput(f'pip install {" ".join(self.pkgs)}'))

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

def setup_package():
    setup(name="spacy_universal_sentence_encoder",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    package_data={'spacy_universal_sentence_encoder': ['meta/*.json']},
    include_package_data=True,
    cmdclass={'install': PostInstall})

if __name__ == "__main__":
    setup_package()