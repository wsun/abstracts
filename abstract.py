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
        self.__numwords = None
        self.__bow = None
        self.__bigram = None
        self.__tfidfbow = None
        self.__tfidfbigram = None

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
        elif attrname == 'numwords': return self.__numwords
        elif attrname == 'bow': return self.__bow
        elif attrname == 'bigram': return self.__bigram
        elif attrname == 'tfidfbow': return self.__tfidfbow
        elif attrname == 'tfidfbigram': return self.__tfidfbigram
        else: return None

    # Set information by name.
    def Set( self, name, val ):
        attrname = name.lower( )
        if attrname == 'path': self.__path = val
        elif attrname == 'title': self.__title = val
        elif attrname == 'text': self.__text = val
        elif attrname == 'tags': self.__tags = val
        elif attrname == 'cleantext': self.__cleantext = val
        elif attrname == 'numwords':self.__numwords = val
        elif attrname == 'bow': self.__bow = val
        elif attrname == 'bigram': self.__bigram = val
        elif attrname == 'tfidfbow': self.__tfidfbow = val
        elif attrname == 'tfidfbigram': self.__tfidfbigram = val
        else:
            print attrname
            raise AttributeError

    # Save a processed abstract.
    # def Save( self ):