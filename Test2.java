package javalang.brewtab.com;

public class VulnerabilityTest {

    public void badMethod() {
        String password = "123456"; 
        System.out.println("Password: " + password);
    }

    public void goodMethod() {
        String password = getPasswordFromUserInput();
        System.out.println("Password: " + password);
    }

    private String getPasswordFromUserInput() {
        return "userInputPassword";
    }

    public static void main(String[] args) {
        VulnerabilityTest test = new VulnerabilityTest();
        test.badMethod();
        test.goodMethod();
    }
}