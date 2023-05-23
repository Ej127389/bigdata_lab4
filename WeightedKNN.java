import java.io.BufferedReader;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.nio.file.FileStore;
import java.nio.file.PathMatcher;
import java.nio.file.WatchService;
import java.nio.file.attribute.UserPrincipalLookupService;
import java.nio.file.spi.FileSystemProvider;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.HashMap;
import java.util.Set;
import java.util.StringTokenizer;
import java.util.Iterator;

import com.google.common.util.concurrent.AtomicDoubleArray;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

public class WeightedKNN {


    public static class KNNMapper
            extends Mapper < Object, Text, IntWritable, Text > {

        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();
        static List < String > test = new ArrayList < String > (); //存储test测试集
        @Override
        protected void setup(Context context) throws IOException, InterruptedException { //获取缓存文件路径的数组
            Path[] paths = DistributedCache.getLocalCacheFiles(context.getConfiguration());
            BufferedReader sb = new BufferedReader(new FileReader(paths[0].toUri().getPath()));
            //读取BufferedReader里面的数据
            String tmp = null;
            while ((tmp = sb.readLine()) != null) {
                test.add(tmp);
            }
            //关闭sb对象
            sb.close();
        }

        //计算加权欧式距离，使用反函数加权即加上1/(1+distance)
        private double Distance(Double[] a, Double[] b) {
            double sum = 0.0;
            for (int i = 0; i < a.length; i++) {
                sum += Math.pow(a[i] - b[i], 2);
            }
            return Math.sqrt(sum) + 1/(1+Math.sqrt(sum));
        }


        public void map(Object key, Text value, Context context) throws IOException,InterruptedException {
            String train[] = value.toString().split(",");
            String lable = train[train.length - 1];
            //训练集由字符格式转化为Double数组
            Double[] train_point = new Double[4];
            for (int i = 0; i < train.length - 1; i++) {
                train_point[i] = Double.valueOf(train[i]);
            }
            //测试集由字符格式转化为Double数组
            for (int i = 0; i < test.size(); i++) {
                String test_poit1[] = test.get(i).toString().split(",");
                Double[] test_poit = new Double[4];
                for (int j = 0; j < test_poit1.length; j++) {
                    test_poit[j] = Double.valueOf(test_poit1[j]);
                }
                //每个测试点的ID作为键key，计算每个测试点与该训练点的距离+"@"+类标签   作为value
                context.write(new IntWritable(i), new Text(String.valueOf(Distance(test_poit, train_point)) + "@" + lable));
            }

        }
    }

    public static class InvertedIndexCombiner extends Reducer < IntWritable, Text, IntWritable, Text > {
        private Text info = new Text();
        int k;
        protected void setup(Context context) throws IOException, InterruptedException {
            Configuration conf=context.getConfiguration();
            k=conf.getInt("K", 1);
        }
        public void reduce(IntWritable key, Iterable < Text > values, Context context) throws IOException, InterruptedException {
            //排序
            TreeMap < Double, String > treemap = new TreeMap < Double, String > ();
            int sum = 0;
            for (Text value: values) {
                String distance_lable[] = value.toString().split("@");
                for (int i = 0; i < distance_lable.length - 1; i = i + 2) {
                    treemap.put(Double.valueOf(distance_lable[i]), distance_lable[i + 1]);
                    //treemap会自动按key升序排序，也就是距离小的排前面
                }
            }
            //得到前k项距离最近
            Iterator < Double > it = treemap.keySet().iterator();
            Map < String, Integer > map = new HashMap < String, Integer > ();
            int num = 0;
            String valueinfo="";
            while (it.hasNext()) {
                Double key1 = it.next();
                valueinfo+=String.valueOf(key1)+"@"+treemap.get(key1)+"@";
                num++;
                if (num > k)
                    break;
            }
            context.write(key,new Text(valueinfo));

        }
    }


    public static class KNNReducer
            extends Reducer < IntWritable, Text, IntWritable, Text > {
        private Text result = new Text();
        int k;
        protected void setup(Context context) throws IOException, InterruptedException {
            Configuration conf=context.getConfiguration();
            k=conf.getInt("K", 1);
        }
        public void reduce(IntWritable key, Iterable < Text > values,Context context) throws IOException,	InterruptedException {
            //排序
            TreeMap < Double, String > treemap = new TreeMap < Double, String > ();
            int sum = 0;
            for (Text val: values) {
                String distance_lable[] = val.toString().split("@");
                for (int i = 0; i < distance_lable.length - 1; i = i + 2) {
                    treemap.put(Double.valueOf(distance_lable[i]), distance_lable[i + 1]);
                    //treemap会自动按key升序排序，也就是距离小的排前面
                }
            }
            //得到前k项距离最近
            Iterator < Double > it = treemap.keySet().iterator();
            Map < String, Integer > map = new HashMap < String, Integer > ();
            int num = 0;
            while (it.hasNext()) {
                Double key1 = it.next();
                if (map.containsKey(treemap.get(key1))) {
                    int temp = map.get(treemap.get(key1));
                    map.put(treemap.get(key1), temp + 1);
                } else {
                    map.put(treemap.get(key1), 1);
                }
                num++;
                if (num > k)
                    break;
            }
            //得到排名最靠前的标签为test的类别
            Iterator < String > it1 = map.keySet().iterator();
            String lable = it1.next();
            int count = map.get(lable);
            while (it1.hasNext()) {
                String now = it1.next();
                if (count < map.get(now)) {
                    lable = now;
                    count = map.get(lable);
                }
            }

            result.set(lable);
            context.write(key, result);
        }
    }




    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        conf.setInt("K",5);
        Job job = Job.getInstance(conf);
        job.setJarByClass(WeightedKNN.class);
        //设置分布式缓存文件
        job.addCacheFile(new URI(args[0]));
        job.setMapperClass(KNNMapper.class);
        job.setCombinerClass(InvertedIndexCombiner.class);
        job.setReducerClass(KNNReducer.class);
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(job, new Path(args[1]));
        FileOutputFormat.setOutputPath(job, new Path(args[2]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}

