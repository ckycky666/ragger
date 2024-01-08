class Measure:
    def __init__(self):

        self.hit1  = 0.0
        self.hit3  = 0.0
        self.hit10 = 0.0
        self.mrr   = 0.0
        self.mr    = 0.0

    def __str__(self) -> str:

        return f'Hit@1={self.hit1}\tHit@3={self.hit3}\tHit@10={self.hit10}\tMR={self.mr}\tMRR={self.mrr}\t'

    def update(self, rank):
        if rank == 1:
            self.hit1 += 1.0
        if rank <= 3:
            self.hit3 += 1.0
        if rank <= 10:
            self.hit10 += 1.0

        self.mr  += rank
        self.mrr += (1.0 / rank)
    
    def normalize(self, num_facts):
            
        self.hit1  /= (num_facts)
        self.hit3  /= (num_facts)
        self.hit10 /= (num_facts)
        self.mr    /= (num_facts)
        self.mrr   /= (num_facts)