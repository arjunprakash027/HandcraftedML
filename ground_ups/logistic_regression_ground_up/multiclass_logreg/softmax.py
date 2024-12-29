import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import kagglehub

# Function to download the dataset, only for Target mutlticlass classification dataset
def get_dataset() -> pd.DataFrame:
    path = kagglehub.dataset_download("mujtabamatin/air-quality-and-pollution-assessment")
    print("Path to dataset files:", path)
    df = pd.read_csv(f"{path}/updated_pollution_dataset.csv")
    return df

def generate_data(
        rows: int,
        cols: int,
        classes: int
) -> pd.DataFrame:

    np.random.seed(42) 
    data = np.random.rand(rows,cols)

    columns = [f"Feature_{i+1}" for i in range(cols)]

    random_classes = [f"Class_{i}" for i in range(1, classes + 1)]
    df = pd.DataFrame(data=data,columns=columns)

    df['Target'] = np.random.choice(random_classes,size=rows)

    return df

def accuracy(actual:list
             ,pred:list) -> float:
    actual = np.array(actual)
    pred = np.array(pred)

    assert len(actual) == len(pred)
    return (np.sum(actual == pred) / len(actual))

def precison_recall(actual:list,
                    pred:list) -> dict:
    actual = np.array(actual)
    pred = np.array(pred)

    assert len(actual) == len(pred)
    
    pr_by_class = {}
    for cl in np.unique(actual):
        tp = np.sum((actual == cl) & (pred == cl))
        fp = np.sum((actual != cl) & (pred == cl))
        fn = np.sum((actual == cl) & (pred != cl))

        print(tp,fp,fn)
        precision = (tp / (tp + fp))
        recall = (tp / (tp + fn))

        pr_by_class[cl] = {
            "precision":precision,
            "recall":recall
        }

    return pr_by_class

class SoftmaxRegression:
    def __init__(self, 
                 learning_rate=0.01, 
                 n_epochs = 1000,
                 l2 = 1) -> None:
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.w = "Weights are not trained yet"
        self.l2 = l2
        self.losses = []
    
    # Function to onehot encode the target variable for mutlticlass computation of gradient decent
    def one_hot_encode(self,
                     y:pd.Series) -> np.ndarray:
        
        return np.eye(y.nunique())[y]
    
    def softmax(self,
                scores:np.ndarray) -> np.ndarray:
        
        # you basically take the maximum of score and substract it with all other scores for numerical stability
        scores -= scores.max()

        # You then take exponent of scores since its negative and sum it along each row 
        # You then divide each row with its transposed exponent to normalize it and transpose it back to give its original shape
        softmax_out = (np.exp(scores).T / np.sum(np.exp(scores), axis=1)).T 

        return softmax_out

    def fit(self,
            X:pd.DataFrame,
            y:pd.Series) -> float:

        self.w = np.zeros([X.shape[1], y.nunique()])

        m = X.shape[0]
        y_mat = self.one_hot_encode(y)
        epsilon = 1e-15

        #print("Shape of x:", X.shape)
        for _ in range(self.n_epochs):
            score = np.dot(X, self.w)

            prob = self.softmax(score)
            
            #print(prob)
            loss = (-1 / m) * np.sum(y_mat * np.log(prob + epsilon)) + ((self.l2/2) * np.sum(self.w * self.w))
            self.losses.append(loss)
            grad = (-1 / m) * np.dot(X.T, (y_mat - prob)) + (self.l2 * self.w)
            
            self.w = self.w - (self.learning_rate * grad)
    
    def predict(self,
                X:pd.DataFrame) -> np.ndarray:
        
        probs = self.softmax(np.dot(X,self.w))
        preds = np.argmax(probs, axis=1)
        return probs, preds
    

if __name__ == '__main__':
    #df = get_dataset()
    df = generate_data(rows=1000,
                       cols=5,
                       classes=4)

    #df = df.rename(columns={'Target':'Target'})

    # Preprocess the target output
    target_list = df['Target'].unique().tolist()
    df['Target'] = df['Target'].apply(lambda x: target_list.index(x))


    Softmax = SoftmaxRegression(learning_rate=1e-5, n_epochs=1000)
    X = df.drop('Target', axis=1)
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    Softmax.fit(X_train, y_train)
    print("Weights",Softmax.w)

    probs_test, preds_test = Softmax.predict(X_test)
    probs_train, preds_train = Softmax.predict(X_train)

    #print(preds)
    #plt.plot(Softmax.losses)
    #plt.show()

    print("Train accuracy:",accuracy(y_train,preds_train))
    print("Test accuracy:",accuracy(y_test,preds_test))

    print("Train PR", precison_recall(y_train,preds_train))
    print("Test PR", precison_recall(y_test,preds_test))