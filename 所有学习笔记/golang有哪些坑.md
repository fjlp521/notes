### golang有哪些坑

#### 1、变量覆盖

```go
var DB *sql.DB	//全局变量DB
func init() {
    DB, _ := sql.Open("mysql", "")	//这里生成的DB为一个新的DB，不是刚才的全局变量DB
}
```

#### 2、指针问题

```go
var head *TreeNode		//这种声明方式 head == nil
head := &TreeNode{}		//这种声明方式 head != nil，head指向一个零值TreeNode（非nil）
head := new(TreeNode)	//这种声明方式 head != nil，head指向一个零值TreeNode（非nil）
```

关于变量修改问题：准则即为，要修改谁，就传谁的指针

对结构体而言，传结构体指针，就可以完成其内部各域的修改

#### 3、二维切片append

```go
nums, num := [][]int{}, []int{}
nums = append(nums, num)	//错误，会覆盖，nums中存的是num的地址，当num变化时，nums也会随之变化
nums = append(nums, append([]int{}, num))	//正确，重新生成一个地址填入nums中
```

#### 4、文件读写操作要注意读写指针位置

- file.Write() 和 file.Read() 会改变读写指针位置；file.WriteAt() 和 file.ReadAt() 不会改变读写指针位置；
- file.Seek()用于设置读写指针位置；
- os.Open()：只读模式；os.Creat()：创建/打开文件，并且清空文件；
- 特别的，用Creat()方式打开文件，写入数据后一定要设置指针位置才能读出数据

### 自定义包导入

- 一个文件夹对应一个包，一个文件夹中**不能有两个同级的go文件分别属于两个不同的包**，当然文件夹中可以套文件夹，在套的文件夹里定义新的包

- **一定要定义go.mod**，此种方法导包最为灵活方便，项目中带着go.mod的话，该项目位置可以随便放

- **三个名称：**go.mod 中的 module 名，新建文件夹的名称，该文件夹下.go文件中的package名， 例如 module aaa，新建文件夹名称为bbb，.go文件中package名为ppp

  ```go
  //方式一：导入和使用
  import "aaa/bbb"	//导入包
  ppp.xxx				//使用包（甚至可以定义为 package 随便，使用时：随便.xxx）
  
  //方式二：导入和使用	
  import t "aaa/bbb"	//导入包
  t.xxx				//使用包
  ```

- 新建文件夹名称即为包名，该文件夹下的.go文件中的package名是调用时使用的，一般这两个名字设置为相同

  
