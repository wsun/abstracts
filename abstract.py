# CS205 Final Project
# Janet Song and Will Sun
#
# Abstract class, represents an abstract.

class Abstract:

    # Construct a new Abstract from the filename provided.

    def __init__( self, filename ):
        self.__path = filename

        #TODO: load file and process
        self.__title = None
        self.__text = None
        self.__tags = None
        self.__cleantext = None
        self.__bow = None
        self.__bownum = None
        self.__bigram = None
        self.__bigramnum = None
        self.__tfidfbow = None
        self.__tfidfbigram = None
        self.__lsa = None
        self.__lda = None
        self.__numtopics = None

    # Print out a Abstract object
    def __repr__( self ):
        return 'Abstract( ' + self.Get( 'title' ) + ' )'
    def __str__( self ):
        return 'Abstract( ' + self.Get( 'title' ) + ' )'

    # Return information by name.
    def Get( self, name ):
        attrname = name.lower( )
        if attrname == 'path': return self.__path
        elif attrname == 'title': return self.__title
        elif attrname == 'text': return self.__text
        elif attrname == 'tags': return self.__tags
        elif attrname == 'cleantext': return self.__cleantext
        elif attrname == 'bow': return self.__bow
        elif attrname == 'bownum': return self.__bownum
        elif attrname == 'bigram': return self.__bigram
        elif attrname == 'bigramnum': return self.__bigramnum
        elif attrname == 'tfidfbow': return self.__tfidfbow
        elif attrname == 'tfidfbigram': return self.__tfidfbigram
        elif attrname == 'lsa': return self.__lsa
        elif attrname == 'lda': return self.__lda
        elif attrname == 'numtopics': return self.__numtopics
        else: return None

    # Set information by name.
    def Set( self, name, val ):
        attrname = name.lower( )
        if attrname == 'path': self.__path = val
        elif attrname == 'title': self.__title = val
        elif attrname == 'text': self.__text = val
        elif attrname == 'tags': self.__tags = val
        elif attrname == 'cleantext': self.__cleantext = val
        elif attrname == 'bow': self.__bow = val
        elif attrname == 'bownum': self.__bownum = val
        elif attrname == 'bigram': self.__bigram = val
        elif attrname == 'bigramnum': self.__bigramnum = val
        elif attrname == 'tfidfbow': self.__tfidfbow = val
        elif attrname == 'tfidfbigram': self.__tfidfbigram = val
        elif attrname == 'lsa': self.__lsa = val
        elif attrname == 'lda': self.__lda = val
        elif attrname == 'numtopics': self.__numtopics = val
        else:
            print attrname
            raise AttributeError

    # Save a processed abstract.
    # def Save( self ):
