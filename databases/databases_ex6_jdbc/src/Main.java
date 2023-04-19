import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.Statement;

public class Main {

	private static Connection conn;
	public static void main(String[] args) throws Exception {
		// Load the SQLite driver
		Class.forName("org.sqlite.JDBC");
		
		// Question 1 - Complete the connection parameters
		// see the documentation for how to make a connection: https://bitbucket.org/xerial/sqlite-jdbc/wiki/Usage
        conn = DriverManager.getConnection("jdbc:sqlite:db/students.db");
        ResultSet rs;
        
        // Question 4 - Insert a student
//         insertStudent("John", "Doe");
        
        // Question 2 - Get all students
        rs = getStudendGrades();
        
        // Question 3 - Find a student
//         rs = searchStudents("ma");
        
        int counter = 0;
        while (rs.next()) {
            System.out.println(rs.getString("studentgrade") + " " + rs.getString("numberStudents"));
            counter++;
        }
        System.out.println("The ResultSet contained " + counter + " results!");
        rs.close();
        conn.close();
	}
	
	private static ResultSet getAllStudents() throws Exception{
		  Statement stat = conn.createStatement();
	      ResultSet rs = stat.executeQuery("SELECT * FROM student");
	      return rs;
	}

	private static ResultSet searchStudents(String name) throws Exception {
		 Statement stat = conn.createStatement();
		 String appendix = "'%".concat(name.toLowerCase()).concat("%'");
		 String query = "SELECT * FROM student WHERE studentfirstname LIKE " + appendix + " OR studentsecondname LIKE " + appendix;
		 System.out.println(query);
	     ResultSet rs = stat.executeQuery(query);
	     return rs;
	}
	
	private static ResultSet insertStudent(String firstname, String lastname) throws Exception {
		PreparedStatement prep = conn.prepareStatement("INSERT into student (studentfirstname, studentsecondname) VALUES (?, ?)");
		prep.setString(1, firstname);
	    prep.setString(2, lastname);
	    prep.executeUpdate();
	    return null;
	}
	private static ResultSet getStudendGrades() throws Exception{
		  Statement stat = conn.createStatement();
	      ResultSet rs = stat.executeQuery("SELECT studentgrade, COUNT(studentid) as numberStudents from student"
	      		+ "GROUP BY studentgrade HAVING studentgrade NOTNULL");
	      return rs;
	}

}
