import pandas as pd
import numpy as np
import time

class ModelData:
    """
    模型使用的训练、检验、测试数据
    """

    def __init__(self,rating_file_path):
        """
        :param rating_file_path:数据集ml-25m的路径ratings.csv
        """
        self.df = pd.read_csv(rating_file_path)
        self.tran_data,self.dev_data,self.test_data = self.load_model_data()

    def load_model_data(self):
        """
        按时间段切分为训练、开发、测试数据集，去掉rating字段，转换为session数据
        :return:
        """
        #train_data = pd.read_csv('../preprocess/data/ratings.csv')
        all_data = self.df.drop(['rating'],axis=1)

        # filter data when user ratings between 5 and 101
        #
        print(len(all_data))
        #filter train data when time from 2008 to 2013
        start_date = '2008-01-01'
        end_date = '2013-03-31'
        start_date_timestamp = int(time.mktime(time.strptime(start_date,'%Y-%m-%d')))
        end_date_timestamp = int(time.mktime(time.strptime(end_date,'%Y-%m-%d')))

        train_data = all_data.loc[(all_data['timestamp'] >= start_date_timestamp) &
                                  (all_data['timestamp'] <= end_date_timestamp)]
        train_data = train_data.groupby('userId').filter(lambda x: len(x) > 5 and len(x) < 101)

        # filter dev data when time from 2013 to 2014
        start_date = '2013-04-01'
        end_date = '2014-04-01'
        start_date_timestamp = int(time.mktime(time.strptime(start_date, '%Y-%m-%d')))
        end_date_timestamp = int(time.mktime(time.strptime(end_date, '%Y-%m-%d')))

        dev_data = all_data.loc[(all_data['timestamp'] >= start_date_timestamp) &
                                (all_data['timestamp'] <= end_date_timestamp)]
        dev_data = dev_data.groupby('userId').filter(lambda x: len(x) > 5 and len(x) < 101)

        # filter test data when time from 2014 to 2015
        start_date = '2014-04-02'
        end_date = '2015-04-01'
        start_date_timestamp = int(time.mktime(time.strptime(start_date, '%Y-%m-%d')))
        end_date_timestamp = int(time.mktime(time.strptime(end_date, '%Y-%m-%d')))

        test_data = all_data.loc[(all_data['timestamp'] >= start_date_timestamp) &
                                 (all_data['timestamp'] <= end_date_timestamp)]
        test_data = test_data.groupby('userId').filter(lambda x: len(x) > 5 and len(x) < 101)

        column_names = ['session_id', 'item_id', 'time']
        train_data.columns = column_names
        dev_data.columns = column_names
        test_data.columns = column_names

        return train_data,dev_data,test_data

class SessionData:
    """
    session数据，添加属性
    session_idx：session id数组
    click_offset:所有session的起始位置
    item_id2index_map:item_id和自然序号的映射
    """

    def __init__(self,data):
        self.data = data
        self.data.sort_values(['session_id','time'],inplace=True)
        self.item_id2index_map = None
        self.add_item_indice()
        self.session_idx = self.get_session_idx()
        self.click_offset = self.get_click_offset()

    def add_item_indice(self,item_map=None):
        """
        在数据集中添加item_id对应的自然序列号
        :param item_map:
        :return:
        """

        if item_map is None:
            item_ids = self.data.item_id.unique()
            item_idx = np.arange(len(item_ids))
            self.item_id2index_map = pd.DataFrame({'item_id':item_ids,'item_idx':item_idx})
        else:
            self.item_id2index_map = item_map
        self.data = pd.merge(self.data,self.item_id2index_map,on='item_id',how='inner')

    def get_session_idx(self):
        return np.arange(self.data.session_id.nunique())

    def get_click_offset(self):
        offset = np.zeros(self.data.session_id.nunique() + 1,dtype=np.int32)
        offset[1:] = self.data.groupby('session_id').size().cumsum()

        return offset

class SessionDataLoader():
    """
    session data的迭代器，用于提取模型训练使用的session数据
    """

    def __init__(self, dataset, batch_size=50):
        """

        :param dataset:
        :param batch_size: 模型训练的item_id的个数，每个session提取一个item_id，共batch_size个session
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.done_sessions_count = 0

    def __iter__(self):
        """
        迭代返回模型训练的input，output，mask
        :return:
        input  模型输入序列，为batch_size个session中的item_id
        output 真实输出，数据是同input相同session的下一个item_id
        mask 保存已完成迭代的session位置，加入新的session去迭代，同时用于重置模型的hidden state,
        """
        mask = []
        finish = False
        data = self.dataset.data
        click_offset = self.dataset.click_offset
        #session_idx = self.dataset.session_idx
        iters = np.arange(self.batch_size)
        max_iter = max(iters)
        start = click_offset[iters]
        end = click_offset[iters]

        while not finish:
            #选取session会话长度最小的会话，保证所有的input和output都在同一会话内，output为input的下一item
            session_min_length = min(end - start)
            target_data = data.item_idx.values[start]
            for i in range(session_min_length - 1):
                input_data = target_data
                target_data = data.item_idx.values[start + i + 1]
                yield input_data,target_data,mask

            start = start + (session_min_length - 1)
            # mask 用来记录已经迭代完成的session，在模型训练时重置对应的session的hidden state为0
            mask = np.arange(self.batch_size)[(end - start <= 1)]
            self.done_sessions_count = len(mask)
            for i in mask:
                max_iter += 1
                if(max_iter >= len(click_offset) - 1):
                    finish = True
                    break
                #更新session迭代完成的idx为新的session idx，直到完成所有session
                iters[i] = max_iter
                start[i] = click_offset[max_iter]
                end[i] = click_offset[max_iter + 1]

    def item_num(self):
        """

        :return: item的总数
        """
        return len(self.dataset.item_id2index_map)

if __name__ == '__main__':
    model_data = ModelData('../preprocess/data/ratings.csv')
    print(model_data.tran_data.head())
    print(len(model_data.dev_data))
    print(len(model_data.test_data))

    session_data = SessionData(model_data.tran_data)
    print(session_data.session_idx[:10])
    print(session_data.click_offset[:10])
    print(session_data.item_id2index_map.head())
    #start_date = '2008-01-01'
    #time_array = time.strptime(start_date, '%Y-%m-%d')
    # 转换为时间戳
    #time_stamp = int(time.mktime(time_array))
    #print(time_stamp)

    session_data_loader = SessionDataLoader(session_data)
    for input_data,target_data,mask in session_data_loader:
        print(input_data)
        print(target_data)
        print(mask)
