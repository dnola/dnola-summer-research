__author__ = 'davidnola'

class PassthroughModel:
    def fit(self, features, targets):
        best_score = 999999999999999999

        for i in range(len(features[0])):
            curscore = 0
            for j in range(len(features)):
                curscore += (targets[j]-features[j][i])
            if curscore < best_score:
                best_score = curscore
                self.best = i

    def predict(self, features):
        print "Best is:", self.best
        return [f[self.best] for f in features]